from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class Batch:
    q: np.ndarray                    # (batch, seq_len, d_head)
    k: np.ndarray                    # (batch, seq_len, d_head)
    v: np.ndarray                    # (batch, seq_len, d_head)
    logit_scales: list[float]        # scales to sweep
    causal_mask: Optional[np.ndarray]  # (seq_len, seq_len) bool or None


def generate(seed: int = 0) -> Batch:
    """
    Deterministic synthetic attention inputs.
    Same seed → same Batch.
    """
    rng = np.random.default_rng(seed)

    batch = 4
    seq_len = 16
    d_head = 32

    # Random unit-norm vectors for q, k, v
    q = rng.normal(size=(batch, seq_len, d_head)).astype(np.float32)
    k = rng.normal(size=(batch, seq_len, d_head)).astype(np.float32)
    v = rng.normal(size=(batch, seq_len, d_head)).astype(np.float32)

    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)
    k = k / (np.linalg.norm(k, axis=-1, keepdims=True) + 1e-8)
    v = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    # Canonical sweep: linear → transition → saturated
    logit_scales = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

    # Optional causal mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len, seq_len), dtype=bool))

    return Batch(
        q=q, k=k, v=v,
        logit_scales=logit_scales,
        causal_mask=causal_mask
    )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def _entropy(p: np.ndarray, axis: int = -1) -> np.ndarray:
    # p: probabilities, shape (..., seq_len)
    return -np.sum(p * np.log(p + 1e-12), axis=axis)


def _reference_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                         logit_scale: float, causal_mask: Optional[np.ndarray]
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute exact attention weights and per-query entropy for given scale.
    Returns (attn_weights, attn_entropy).
    Shapes: attn_weights (batch, seq_len, seq_len), attn_entropy (batch, seq_len)
    """
    batch, seq_len, d_head = q.shape
    logits = np.einsum('bqd,bkd->bqk', q, k) * logit_scale  # (batch, seq_len, seq_len)

    if causal_mask is not None:
        logits = np.where(causal_mask, logits, -1e9)

    attn_weights = _softmax(logits, axis=-1)  # (batch, seq_len, seq_len)
    attn_entropy = _entropy(attn_weights, axis=-1)  # (batch, seq_len)
    return attn_weights, attn_entropy


def evaluate(model_fn) -> dict:
    """
    Run model_fn over each logit_scale in the batch, collect payload.
    model_fn signature:
        model_fn(q, k, v, logit_scale, causal_mask) -> dict with keys:
            'attn_weights': (batch, seq_len, seq_len)
            'attn_entropy': (batch, seq_len)
            'saturation_score': float
    Returns payload dict matching benchmark.score contract.
    """
    batch = generate(seed=0)  # canonical seed
    sweep_records = []

    for scale in batch.logit_scales:
        out = model_fn(batch.q, batch.k, batch.v, scale, batch.causal_mask)

        # Validate required keys
        for k in ('attn_weights', 'attn_entropy', 'saturation_score'):
            if k not in out:
                raise KeyError(f"model_fn output missing key: {k}")

        attn_w = np.asarray(out['attn_weights'], dtype=np.float32)
        attn_ent = np.asarray(out['attn_entropy'], dtype=np.float32)
        sat_score = float(out['saturation_score'])

        # Reference (ground truth) for this scale
        ref_w, ref_ent = _reference_attention(batch.q, batch.k, batch.v, scale, batch.causal_mask)

        record = {
            'logit_scale': float(scale),
            'attn_weights': attn_w,
            'attn_entropy': attn_ent,
            'saturation_score': sat_score,
            # Mean over (batch, query) of the per-query max attention weight.
            # NB: a plain global .max() is pinned to 1.0 by the causal mask
            # (query 0 has a single valid key), so we average per-query maxima
            # to get a value that actually rises with saturation.
            'max_attn_weight': float(attn_w.max(axis=-1).mean()),
            'mean_entropy': float(attn_ent.mean()),
            # Reference (analytic ground-truth) fields, consumed by benchmark.score.
            'ref_max_attn_weight': float(ref_w.max(axis=-1).mean()),
            'ref_mean_entropy': float(ref_ent.mean()),
        }
        sweep_records.append(record)

    payload = {
        'version': 1,
        'sweep': sweep_records,
        'config': {
            'batch': batch.q.shape[0],
            'seq_len': batch.q.shape[1],
            'd_head': batch.q.shape[2],
            'seed': 0,
        }
    }
    return payload


def random_model_fn():
    """
    Returns a model_fn that outputs random/zero values of the correct shape.
    Pure NumPy, no torch, no GPU. Used for smoke test.
    """
    def _fn(q: np.ndarray, k: np.ndarray, v: np.ndarray,
            logit_scale: float, causal_mask: Optional[np.ndarray]) -> dict:
        batch, seq_len, _ = q.shape
        # Uniform attention weights (max entropy)
        attn_weights = np.ones((batch, seq_len, seq_len), dtype=np.float32) / seq_len
        if causal_mask is not None:
            attn_weights = np.where(causal_mask, attn_weights, 0.0)
            # Renormalize
            attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-12)
        attn_entropy = np.full((batch, seq_len), np.log(seq_len), dtype=np.float32)
        # Random saturation score in [0, 1]
        saturation_score = float(np.random.default_rng().random())
        return {
            'attn_weights': attn_weights,
            'attn_entropy': attn_entropy,
            'saturation_score': saturation_score,
        }
    return _fn