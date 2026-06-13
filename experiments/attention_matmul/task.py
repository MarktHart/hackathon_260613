"""Task for the `attention_matmul` goal.

Synthetic generator + evaluator for query-key-value attribution. The data and
the scoring contract live here so that every attempt imports `generate`,
`evaluate`, and `random_model_fn` instead of duplicating them.

Pure NumPy. No I/O, no network, no torch, no GPU.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, List

# ----------------------------------------------------------------------
# Canonical configuration (frozen). See README.md.
# ----------------------------------------------------------------------
CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "d_head": 16,
    "seq_len": 32,
    "batch_size": 16,
}

# Sweep axis: query-key alignment regime.
CONDITIONS: List[str] = ["orthogonal", "cos_0p3", "cos_0p7", "uniform"]

# Canonical headline condition.
CANONICAL_CONDITION = "cos_0p3"

# Fixed evaluation seed; `generate` is deterministic for any seed.
EVAL_SEED = 0

# model_fn(Q, K, V) -> attribution (batch, n_heads, seq_len, seq_len)
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    """One condition's Q, K, V plus the ground-truth attention and output."""
    Q: np.ndarray            # (batch, n_heads, seq_len, d_head)
    K: np.ndarray            # (batch, n_heads, seq_len, d_head)
    V: np.ndarray            # (batch, n_heads, seq_len, d_head)
    true_attn: np.ndarray    # (batch, n_heads, seq_len, seq_len) softmax(QK^T/√d)
    true_output: np.ndarray  # (batch, n_heads, seq_len, d_head) true_attn @ V
    condition: str           # one of CONDITIONS


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def _generate_condition(seed: int, cond_index: int, condition: str) -> Batch:
    """Deterministic batch for one condition (no string hashing)."""
    rng = np.random.default_rng(
        (int(seed) * 1_000_003 + cond_index * 9_973 + 17) & 0x7FFFFFFF
    )
    B, H, T, D = (CONFIG["batch_size"], CONFIG["n_heads"],
                  CONFIG["seq_len"], CONFIG["d_head"])

    # Values are always i.i.d. Gaussian.
    V = rng.standard_normal((B, H, T, D)).astype(np.float32)

    if condition == "orthogonal":
        # Each query attends to exactly one key via an orthonormal basis.
        Q = np.zeros((B, H, T, D), dtype=np.float32)
        K = np.zeros((B, H, T, D), dtype=np.float32)
        for b in range(B):
            for h in range(H):
                basis, _ = np.linalg.qr(rng.standard_normal((D, D)))
                for i in range(T):
                    key_idx = (i + h) % T
                    vec = basis[:, i % D].astype(np.float32)
                    Q[b, h, i] = vec
                    K[b, h, key_idx] = vec

    elif condition in ("cos_0p3", "cos_0p7"):
        target_cos = 0.3 if condition == "cos_0p3" else 0.7
        Q = rng.standard_normal((B, H, T, D)).astype(np.float32)
        K = rng.standard_normal((B, H, T, D)).astype(np.float32)
        Q = Q / (np.linalg.norm(Q, axis=-1, keepdims=True) + 1e-8)
        K = K / (np.linalg.norm(K, axis=-1, keepdims=True) + 1e-8)
        for b in range(B):
            for h in range(H):
                for i in range(T):
                    key_idx = (i + h) % T
                    q = Q[b, h, i]
                    k = K[b, h, key_idx]
                    orth = k - np.dot(k, q) * q
                    nrm = np.linalg.norm(orth)
                    if nrm > 1e-8:
                        orth = orth / nrm
                    else:
                        orth = rng.standard_normal(D).astype(np.float32)
                        orth = orth / (np.linalg.norm(orth) + 1e-8)
                    K[b, h, key_idx] = (
                        target_cos * q
                        + np.sqrt(max(0.0, 1.0 - target_cos ** 2)) * orth
                    ).astype(np.float32)

    elif condition == "uniform":
        Q = rng.standard_normal((B, H, T, D)).astype(np.float32)
        K = rng.standard_normal((B, H, T, D)).astype(np.float32)

    else:
        raise ValueError(f"Unknown condition: {condition!r}")

    scale = 1.0 / np.sqrt(D)
    scores = np.einsum("bhid,bhjd->bhij", Q, K) * scale
    true_attn = _softmax(scores, axis=-1)
    true_output = np.einsum("bhij,bhjd->bhid", true_attn, V)

    return Batch(
        Q=Q, K=K, V=V,
        true_attn=true_attn.astype(np.float32),
        true_output=true_output.astype(np.float32),
        condition=condition,
    )


def generate(seed: int = 0) -> List[Batch]:
    """Deterministic list of one Batch per condition for a given seed."""
    return [
        _generate_condition(seed, i, cond)
        for i, cond in enumerate(CONDITIONS)
    ]


# ----------------------------------------------------------------------
# Metric helpers (reductions live here; only scalars enter the payload)
# ----------------------------------------------------------------------
def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Mean over rows of KL(p || q). p, q are (..., T); q is renormalised."""
    p = np.clip(np.asarray(p, dtype=np.float64), eps, None)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, None)
    q = q / np.sum(q, axis=-1, keepdims=True)
    kl = np.sum(p * np.log(p / q), axis=-1)
    return float(np.mean(kl))


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((np.asarray(a, dtype=np.float64)
                          - np.asarray(b, dtype=np.float64)) ** 2))


def _rowsum_mae(attrib: np.ndarray) -> float:
    rowsums = np.sum(np.asarray(attrib, dtype=np.float64), axis=-1)
    return float(np.mean(np.abs(rowsums - 1.0)))


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """Run `model_fn` over every condition and return a benchmark-ready payload.

    The payload holds only scalars and small dicts — no raw tensors.
    """
    batches = generate(seed=EVAL_SEED)
    B, H, T, D = (CONFIG["batch_size"], CONFIG["n_heads"],
                  CONFIG["seq_len"], CONFIG["d_head"])
    expected_shape = (B, H, T, T)

    sweep: List[Dict[str, Any]] = []
    linear_baseline: Dict[str, Dict[str, float]] = {}

    for batch in batches:
        pred = np.asarray(model_fn(batch.Q, batch.K, batch.V), dtype=np.float64)
        if pred.shape != expected_shape:
            raise ValueError(
                f"model_fn returned shape {pred.shape}, expected {expected_shape}"
            )

        pred_output = np.einsum("bhij,bhjd->bhid", pred, batch.V)

        sweep.append({
            "qk_alignment": batch.condition,
            "output_mse": _mse(pred_output, batch.true_output),
            "attribution_kl": _kl_divergence(batch.true_attn, pred),
            "rowsum_mae": _rowsum_mae(pred),
        })

        # Fixed strawman: uniform attribution over keys.
        uniform = np.full(expected_shape, 1.0 / T, dtype=np.float64)
        uniform_output = np.einsum("bhij,bhjd->bhid", uniform, batch.V)
        linear_baseline[batch.condition] = {
            "output_mse": _mse(uniform_output, batch.true_output),
            "attribution_kl": _kl_divergence(batch.true_attn, uniform),
            "rowsum_mae": _rowsum_mae(uniform),
        }

    return {
        "version": 1,
        "model_name": "synthetic_attention_matmul",
        "config": dict(CONFIG),
        "canonical_condition": CANONICAL_CONDITION,
        "conditions": list(CONDITIONS),
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


# ----------------------------------------------------------------------
# Random reference model (smoke test)
# ----------------------------------------------------------------------
def random_model_fn() -> ModelFn:
    """A model_fn with the real signature that emits uniform attribution.

    Rows sum to 1 by construction. Pure NumPy; no torch, no GPU.
    """
    def _random_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        B, H, T, _ = np.asarray(Q).shape
        return np.full((B, H, T, T), 1.0 / T, dtype=np.float32)

    return _random_fn
