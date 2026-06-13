import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    """Container for a single evaluation batch (one query + three keys)."""
    query: np.ndarray          # shape (d_model,)
    keys: np.ndarray           # shape (3, d_model)
    alpha: float               # sweep parameter for this batch


# Fixed random embeddings for reproducibility
_EMBEDDING_SEED = 42
_D_MODEL = 32
_TOKEN_TYPES = ["TARGET", "DISTRACTOR_A", "DISTRACTOR_B", "DISTRACTOR_C"]
_NUM_DISTRACTORS = 3

# Pre-computed embeddings (deterministic)
_rng_embed = np.random.default_rng(_EMBEDDING_SEED)
_TOKEN_EMBEDDINGS = {
    tok: _rng_embed.normal(size=_D_MODEL).astype(np.float32)
    for tok in _TOKEN_TYPES
}
# Noise vector for query construction.
# Orthogonalize against the TARGET embedding ONLY, so that alpha cleanly
# controls target similarity: dot(noise, e_TARGET) == 0, hence at alpha=0 the
# query has exactly zero target similarity. The noise INTENTIONALLY retains
# small incidental similarity to the distractors -- that spurious match is the
# very thing the attention head must avoid collapsing onto. (Sequential
# Gram-Schmidt against all four non-orthogonal embeddings would NOT yield a
# vector orthogonal to all of them; it would only be orthogonal to the last.)
_NOISE_VEC = _rng_embed.normal(size=_D_MODEL).astype(np.float32)
_e_target = _TOKEN_EMBEDDINGS["TARGET"]
_NOISE_VEC -= _e_target * (np.dot(_NOISE_VEC, _e_target) / np.dot(_e_target, _e_target))
_NOISE_VEC = _NOISE_VEC / np.linalg.norm(_NOISE_VEC)

# Distractor embeddings (target never appears in keys)
_DISTRACTOR_EMBEDS = np.stack([
    _TOKEN_EMBEDDINGS["DISTRACTOR_A"],
    _TOKEN_EMBEDDINGS["DISTRACTOR_B"],
    _TOKEN_EMBEDDINGS["DISTRACTOR_C"],
], axis=0)  # shape (3, d_model)


# Canonical sweep: 11 points from 0.0 to 1.0 inclusive
CANONICAL_SWEEP_ALPHAS = [round(i * 0.1, 1) for i in range(11)]


def generate(seed: int = 0) -> list[Batch]:
    """
    Generate the full sweep of batches. Deterministic: same seed → same batches.
    The seed only affects the query noise realization (alpha > 0); alpha=0 is seed-independent.
    """
    rng = np.random.default_rng(seed)
    batches = []
    for alpha in CANONICAL_SWEEP_ALPHAS:
        # Query = alpha * e_TARGET + (1-alpha) * e_NOISE (orthogonal to all tokens)
        query_vec = alpha * _TOKEN_EMBEDDINGS["TARGET"] + (1 - alpha) * _NOISE_VEC
        # Add tiny seed-dependent jitter for alpha > 0 to make batches vary with seed
        if alpha > 0:
            jitter = rng.normal(scale=1e-6, size=_D_MODEL).astype(np.float32)
            query_vec = query_vec + jitter
            query_vec = query_vec / np.linalg.norm(query_vec)
        batches.append(Batch(query=query_vec.astype(np.float32),
                             keys=_DISTRACTOR_EMBEDS.astype(np.float32),
                             alpha=alpha))
    return batches


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def _linear_attention(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Linear attention baseline: weights ∝ query·key, renormalized to sum=1."""
    scores = keys @ query  # shape (3,)
    # Shift to non-negative for interpretability (doesn't change softmax)
    scores = scores - np.min(scores) + 1e-8
    return scores / np.sum(scores)


def evaluate(model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> dict:
    """
    Run model_fn over the canonical sweep, return payload for benchmark.score.

    Args:
        model_fn: Callable with signature (query: (d_model,), keys: (3, d_model))
                  -> attn_weights: (3,), non-negative, sums to 1.

    Returns:
        Payload dict matching the contract in README.md.
    """
    batches = generate(seed=0)  # canonical seed for evaluation
    sweep_records = []

    for batch in batches:
        attn_weights = model_fn(batch.query, batch.keys)
        # Validate output
        attn_weights = np.asarray(attn_weights, dtype=np.float32)
        if attn_weights.shape != (3,):
            raise ValueError(f"model_fn returned shape {attn_weights.shape}, expected (3,)")
        if not np.all(attn_weights >= -1e-6):
            raise ValueError("model_fn returned negative attention weights")
        if abs(np.sum(attn_weights) - 1.0) > 1e-4:
            raise ValueError(f"attention weights sum to {np.sum(attn_weights):.6f}, expected 1.0")

        # Per-slice measurements
        max_weight = float(np.max(attn_weights))
        # Entropy in nats
        entropy = float(-np.sum(attn_weights * np.log(attn_weights + 1e-12)))
        # KL divergence from uniform (1/3, 1/3, 1/3)
        uniform = 1.0 / 3.0
        uniform_kl = float(np.sum(attn_weights * np.log((attn_weights + 1e-12) / uniform)))

        sweep_records.append({
            "alpha": batch.alpha,
            "query": batch.query.tolist(),
            "keys": batch.keys.tolist(),
            "attn_weights": attn_weights.tolist(),
            "max_weight": max_weight,
            "entropy": entropy,
            "uniform_kl": uniform_kl,
        })

    return {
        "version": 1,
        "d_model": _D_MODEL,
        "sweep": sweep_records,
        "sweep_alphas": CANONICAL_SWEEP_ALPHAS,
    }


def random_model_fn() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a model_fn that outputs uniform attention (the minimax optimum at alpha=0).
    Pure NumPy, no torch, no GPU. Used for smoke test.
    """
    def _fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        return np.array([1.0/3.0, 1.0/3.0, 1.0/3.0], dtype=np.float32)
    return _fn