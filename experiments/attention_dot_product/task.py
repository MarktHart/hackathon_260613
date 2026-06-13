"""Task for the `attention_dot_product` goal.

Pure NumPy, deterministic, no I/O, no torch, no GPU.

Exports:
    generate(seed) -> Batch
    evaluate(model_fn) -> payload dict   (shape consumed by benchmark.score)
    random_model_fn() -> ModelFn         (smoke-test stand-in)

The goal asks whether an attempt's `model_fn` reproduces *scaled dot-product
attention* — softmax(Q Kᵀ / sqrt(d_head)) V — and how that fidelity holds up
as the sequence length (and therefore the softmax competition) grows.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

# The goal's contract with attempts.  Q, K, V are each
# (batch, n_heads, seq_len, d_head); the return is the attention output with
# the SAME shape as V.
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# ----------------------------------------------------------------------------
# Canonical configuration
# ----------------------------------------------------------------------------
_D_HEAD = 16
_N_HEADS = 4
_BATCH_SIZE = 8
_CANONICAL_SEQ_LEN = 32
_SEQ_LEN_SWEEP = [8, 16, 32, 64, 128]
_GEN_SEED = 0


def config() -> dict:
    """Self-describing canonical configuration (also embedded in the payload)."""
    return {
        "d_head": _D_HEAD,
        "n_heads": _N_HEADS,
        "batch_size": _BATCH_SIZE,
        "canonical_seq_len": _CANONICAL_SEQ_LEN,
        "seq_len_sweep": list(_SEQ_LEN_SWEEP),
    }


@dataclass(frozen=True)
class Batch:
    """One evaluation batch at a fixed sequence length."""
    seq_len: int
    Q: np.ndarray       # (batch, n_heads, seq_len, d_head)
    K: np.ndarray       # (batch, n_heads, seq_len, d_head)
    V: np.ndarray       # (batch, n_heads, seq_len, d_head)
    gt_out: np.ndarray  # (batch, n_heads, seq_len, d_head) — true attention output


def _attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Reference scaled dot-product attention: softmax(QKᵀ/sqrt(d)) V."""
    d_head = Q.shape[-1]
    scale = 1.0 / np.sqrt(d_head)
    scores = np.einsum("bhsd,bhtd->bhst", Q, K) * scale
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    return np.einsum("bhst,bhtd->bhsd", weights, V)


def _generate_seq(seq_len: int, seed: int) -> Batch:
    rng = np.random.default_rng(seed)
    shape = (_BATCH_SIZE, _N_HEADS, seq_len, _D_HEAD)
    Q = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    K = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    V = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    gt_out = _attention(Q, K, V).astype(np.float32)
    return Batch(seq_len=seq_len, Q=Q, K=K, V=V, gt_out=gt_out)


def generate(seed: int = 0) -> Batch:
    """Deterministic canonical batch (seq_len = canonical_seq_len).

    Same seed -> same batch.  The sweep batches are generated internally by
    `evaluate`; this returns only the canonical condition for callers that want
    to inspect the data directly.
    """
    return _generate_seq(_CANONICAL_SEQ_LEN, seed)


def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """MSE, relative Frobenius error and mean per-token cosine similarity."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    mse = float(np.mean((pred - gt) ** 2))

    diff_norm = float(np.linalg.norm(pred - gt))
    gt_norm = float(np.linalg.norm(gt))
    rel_error = diff_norm / gt_norm if gt_norm > 0.0 else 0.0

    pred_flat = pred.reshape(-1, pred.shape[-1])
    gt_flat = gt.reshape(-1, gt.shape[-1])
    pred_n = np.linalg.norm(pred_flat, axis=-1)
    gt_n = np.linalg.norm(gt_flat, axis=-1)
    dots = np.sum(pred_flat * gt_flat, axis=-1)
    mask = (pred_n > 0) & (gt_n > 0)
    cos = np.zeros_like(dots)
    cos[mask] = dots[mask] / (pred_n[mask] * gt_n[mask])
    cos_sim = float(np.mean(cos)) if cos.size else 0.0

    return {"mse": mse, "rel_error": float(rel_error), "cos_sim": cos_sim}


def _baseline_mse(batch: Batch) -> float:
    """Strawman: uniform attention (mean of V over keys), no dot-product."""
    uniform = np.mean(batch.V, axis=2, keepdims=True)              # (b, h, 1, d)
    uniform = np.broadcast_to(uniform, batch.V.shape)             # (b, h, s, d)
    gt = np.asarray(batch.gt_out, dtype=np.float64)
    return float(np.mean((uniform.astype(np.float64) - gt) ** 2))


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the seq_len sweep and return the scoring payload."""
    sweep = []
    for seq_len in _SEQ_LEN_SWEEP:
        batch = _generate_seq(seq_len, seed=_GEN_SEED)
        pred = np.asarray(model_fn(batch.Q, batch.K, batch.V), dtype=np.float32)
        if pred.shape != batch.gt_out.shape:
            raise ValueError(
                f"model_fn returned shape {pred.shape}, "
                f"expected {batch.gt_out.shape} at seq_len={seq_len}"
            )
        m = _metrics(pred, batch.gt_out)
        sweep.append({
            "seq_len": int(seq_len),
            "mse": m["mse"],
            "rel_error": m["rel_error"],
            "cos_sim": m["cos_sim"],
            "baseline_mse": _baseline_mse(batch),
        })

    return {
        "version": 1,
        "model_name": "synthetic_scaled_dot_product_attention",
        "config": config(),
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """Return a stand-in `model_fn` that emits random output of the right shape.

    Same signature as a real attempt's model_fn; used by the smoke test.
    """
    def _fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.normal(0.0, 1.0, size=np.asarray(V).shape).astype(np.float32)

    return _fn
