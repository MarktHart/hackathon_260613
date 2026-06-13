import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    queries: np.ndarray          # [n_queries, d_model]
    keys: np.ndarray             # [n_keys, d_model]
    scales: np.ndarray           # [n_conditions]
    tail_types: list[str]        # length n_conditions
    alphas: list[float | None]   # length n_conditions
    rates: list[float | None]    # length n_conditions
    condition_ids: list[str]     # length n_conditions

ModelFn = Callable[[np.ndarray, np.ndarray, float], np.ndarray]

CANONICAL_N_QUERIES = 32
CANONICAL_N_KEYS = 128
CANONICAL_D_MODEL = 64
CANONICAL_SEED = 42

# Sweep: 5 Pareto alphas + 4 Exponential rates
PARETO_ALPHAS = [0.1, 0.3, 0.5, 0.7, 1.0]
EXP_RATES = [0.5, 1.0, 2.0, 5.0]

# Per-condition temperature (logit scale) — the only per-condition signal
# model_fn receives. Heavier tail (smaller Pareto alpha) -> higher temperature
# -> sharper softmax; lighter tail (higher Exponential rate) -> lower
# temperature -> flatter softmax. Aligned element-wise with the lists above.
PARETO_SCALES = [2.8, 2.4, 2.0, 1.6, 1.0]   # aligned with PARETO_ALPHAS
EXP_SCALES = [0.8, 0.5, 0.3, 0.15]          # aligned with EXP_RATES

def generate(seed: int = CANONICAL_SEED) -> Batch:
    """Deterministic synthetic data for the attention quantile sweep."""
    rng = np.random.default_rng(seed)
    
    # Fixed random unit vectors for queries and keys
    queries = rng.normal(size=(CANONICAL_N_QUERIES, CANONICAL_D_MODEL)).astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    
    keys = rng.normal(size=(CANONICAL_N_KEYS, CANONICAL_D_MODEL)).astype(np.float32)
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    
    n_conditions = len(PARETO_ALPHAS) + len(EXP_RATES)
    # Temperature varies per condition (see PARETO_SCALES / EXP_SCALES). The
    # order here MUST match the condition-construction order below (pareto
    # block first, then exponential block).
    assert len(PARETO_SCALES) == len(PARETO_ALPHAS)
    assert len(EXP_SCALES) == len(EXP_RATES)
    scales = np.array(PARETO_SCALES + EXP_SCALES, dtype=np.float32)
    
    tail_types = []
    alphas = []
    rates = []
    condition_ids = []
    
    for a in PARETO_ALPHAS:
        tail_types.append("pareto")
        alphas.append(a)
        rates.append(None)
        condition_ids.append(f"pareto_{a:.1f}".replace(".", "p"))
    
    for r in EXP_RATES:
        tail_types.append("exponential")
        alphas.append(None)
        rates.append(r)
        condition_ids.append(f"exponential_{r:.1f}".replace(".", "p"))
    
    return Batch(
        queries=queries,
        keys=keys,
        scales=scales,
        tail_types=tail_types,
        alphas=alphas,
        rates=rates,
        condition_ids=condition_ids,
    )

def _compute_attention_weights(queries: np.ndarray, keys: np.ndarray, scale: float,
                                tail_type: str, alpha: float | None, rate: float | None) -> np.ndarray:
    """
    Generate ground-truth attention logits from the specified parametric family,
    then softmax. This is what a 'perfect' model would produce.
    """
    n_q, n_k = queries.shape[0], keys.shape[0]
    logits = np.zeros((n_q, n_k), dtype=np.float32)
    
    if tail_type == "pareto":
        assert alpha is not None
        # Pareto-distributed logits: sample from Pareto(alpha) then shift
        # Use the query-key dot product as a base, then add Pareto noise
        base = queries @ keys.T  # [n_q, n_k] in [-1, 1]
        # Pareto samples: (1 - u)^(-1/alpha) for u ~ Uniform(0,1)
        u = np.random.default_rng(12345).uniform(0, 1, size=(n_q, n_k))
        pareto_samples = (1 - u) ** (-1.0 / alpha)
        # Normalize pareto samples per query to have mean 0, std 1
        pareto_samples = (pareto_samples - pareto_samples.mean(axis=1, keepdims=True)) / (pareto_samples.std(axis=1, keepdims=True) + 1e-8)
        logits = base + pareto_samples
    elif tail_type == "exponential":
        assert rate is not None
        base = queries @ keys.T
        u = np.random.default_rng(12345).uniform(0, 1, size=(n_q, n_k))
        exp_samples = -np.log(u) / rate
        exp_samples = (exp_samples - exp_samples.mean(axis=1, keepdims=True)) / (exp_samples.std(axis=1, keepdims=True) + 1e-8)
        logits = base + exp_samples
    else:
        raise ValueError(f"Unknown tail_type: {tail_type}")
    
    logits = logits * scale
    # Softmax per query
    logits_max = logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    attn = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return attn.astype(np.float32)

def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over all sweep conditions, compute quantile stats, return payload."""
    batch = generate()
    sweep_records = []
    
    for i in range(len(batch.condition_ids)):
        cond_id = batch.condition_ids[i]
        tail_type = batch.tail_types[i]
        alpha = batch.alphas[i]
        rate = batch.rates[i]
        scale = float(batch.scales[i])
        
        # Call the model function
        attn = model_fn(batch.queries, batch.keys, scale)
        
        # Validate output shape
        if attn.shape != (CANONICAL_N_QUERIES, CANONICAL_N_KEYS):
            raise ValueError(f"model_fn returned shape {attn.shape}, expected ({CANONICAL_N_QUERIES}, {CANONICAL_N_KEYS})")
        if not np.allclose(attn.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"model_fn rows don't sum to 1: sums = {attn.sum(axis=1)[:5]}...")
        
        # Flatten all query-key attention weights for quantile computation
        flat = attn.reshape(-1)
        q50 = float(np.percentile(flat, 50))
        q90 = float(np.percentile(flat, 90))
        # Robust denominator. A very sparse / heavy-tailed attention (e.g. a
        # top-k method) can have >=50% of weights exactly zero, making the
        # median weight 0. Rather than emit inf (which payload validation
        # rejects, crashing exactly the heavy-tail methods this goal rewards),
        # floor the denominator at the smallest positive weight so the ratio
        # stays finite and grows monotonically with concentration. Rows sum to
        # 1, so at least one positive weight always exists.
        if q50 > 0:
            denom = q50
        else:
            positive = flat[flat > 0]
            denom = float(positive.min()) if positive.size > 0 else 1.0
        ratio = q90 / denom
        
        sweep_records.append({
            "condition_id": cond_id,
            "tail_type": tail_type,
            "alpha": alpha,
            "rate": rate,
            "quantile_50": q50,
            "quantile_90": q90,
            "quantile_ratio": float(ratio),
        })
    
    return {
        "version": 1,
        "config": {
            "n_queries": CANONICAL_N_QUERIES,
            "n_keys": CANONICAL_N_KEYS,
            "d_model": CANONICAL_D_MODEL,
            "seed": CANONICAL_SEED,
        },
        "sweep": sweep_records,
    }

def random_model_fn() -> ModelFn:
    """Returns a model_fn that produces uniform attention (valid shape, sums to 1)."""
    def _uniform_attn(queries: np.ndarray, keys: np.ndarray, scale: float) -> np.ndarray:
        n_q = queries.shape[0]
        n_k = keys.shape[0]
        return np.full((n_q, n_k), 1.0 / n_k, dtype=np.float32)
    return _uniform_attn