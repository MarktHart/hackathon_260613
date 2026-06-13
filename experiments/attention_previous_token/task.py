"""Task for the attention_previous_token goal.

Synthetic previous-token-attention probe. Exports:

    generate(seed) -> Batch          deterministic residual streams
    evaluate(model_fn) -> dict       runs the head, returns the payload
    random_model_fn() -> ModelFn     a contract-shaped no-op model (zeros)

Pure NumPy. No torch, no GPU, no I/O. The payload returned by ``evaluate``
matches ``benchmark.score``'s contract exactly (see README.md).
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ---- Canonical constants (must match README.md) ----
SEQ_LEN = 64
D = 64
N_SEEDS = 16
CONTENT_SCALE = 0.5
POS_SCALE = 1.0
CANONICAL_NOISE = 0.0
NOISE_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0]

# Fixed RNG seeds so every attempt sees identical data / noise.
_DATA_SEED = 0
_NOISE_SEED = 20240613

# model_fn: (seq_len, d) float32 residual -> (seq_len, seq_len) float32 logits
ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    residuals: np.ndarray  # (n_seeds, seq_len, d) float32 — base residual streams
    seq_len: int
    d: int
    n_seeds: int


def _positional_embeddings(seq_len: int, d: int) -> np.ndarray:
    """Standard sinusoidal positional embeddings, (seq_len, d)."""
    pos = np.arange(seq_len)[:, None].astype(np.float64)
    i = np.arange(d)[None, :].astype(np.float64)
    div = np.power(10000.0, 2.0 * np.floor(i / 2.0) / d)
    angles = pos / div
    emb = np.where(np.arange(d)[None, :] % 2 == 0, np.sin(angles), np.cos(angles))
    return emb.astype(np.float32)


def generate(seed: int = 0) -> Batch:
    """Deterministic batch of residual streams (positional signal + content)."""
    rng = np.random.default_rng(seed)
    pos_emb = _positional_embeddings(SEQ_LEN, D) * POS_SCALE  # (seq_len, d)
    content = rng.normal(size=(N_SEEDS, SEQ_LEN, D)).astype(np.float32) * CONTENT_SCALE
    residuals = (pos_emb[None, :, :] + content).astype(np.float32)
    return Batch(residuals=residuals, seq_len=SEQ_LEN, d=D, n_seeds=N_SEEDS)


def _causal_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax with a causal mask (key j > query i forbidden)."""
    L = logits.shape[0]
    mask = np.triu(np.ones((L, L), dtype=bool), k=1)  # True where j > i
    x = np.where(mask, -1e30, logits.astype(np.float64))
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    e = np.where(mask, 0.0, e)
    denom = e.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return e / denom


def _uniform_prev_baseline(seq_len: int) -> float:
    """Previous-token mass under a content-blind uniform causal head.

    Query i (i>=1) spreads 1/(i+1) over each of its i+1 allowed keys, so the
    previous token gets 1/(i+1). Average over i in [1, seq_len-1].
    """
    return float(np.mean([1.0 / (i + 1) for i in range(1, seq_len)]))


def evaluate(model_fn: ModelFn) -> dict:
    """Run ``model_fn`` over the canonical batch; return the scoring payload."""
    batch = generate(seed=_DATA_SEED)
    L = batch.seq_len
    noise_rng = np.random.default_rng(_NOISE_SEED)
    uniform_baseline = _uniform_prev_baseline(L)

    sweep = []
    for noise in NOISE_SWEEP:
        prevs, selfs, twos = [], [], []
        for s in range(batch.n_seeds):
            resid = batch.residuals[s]
            if noise > 0.0:
                resid = resid + noise_rng.normal(size=resid.shape).astype(np.float32) * noise
            resid = resid.astype(np.float32)

            logits = np.asarray(model_fn(resid), dtype=np.float64)
            if logits.shape != (L, L):
                raise ValueError(
                    f"model_fn must return a ({L}, {L}) array of logits, "
                    f"got shape {logits.shape}"
                )
            if not np.all(np.isfinite(logits)):
                raise ValueError("model_fn returned non-finite logits")

            attn = _causal_softmax(logits)  # (L, L), rows sum to 1
            prevs.append(float(np.diagonal(attn, offset=-1).mean()))   # A[i, i-1], i>=1
            selfs.append(float(np.diagonal(attn, offset=0).mean()))    # A[i, i]
            twos.append(float(np.diagonal(attn, offset=-2).mean()))    # A[i, i-2], i>=2

        sweep.append({
            "noise": float(noise),
            "prev_token_attention": float(np.mean(prevs)),
            "self_attention": float(np.mean(selfs)),
            "two_back_attention": float(np.mean(twos)),
            "uniform_baseline": float(uniform_baseline),
            "n_seeds": int(batch.n_seeds),
        })

    return {
        "version": 1,
        "model_name": "synthetic_previous_token",
        "seq_len": int(L),
        "d": int(batch.d),
        "canonical_noise": float(CANONICAL_NOISE),
        "noise_sweep": [float(n) for n in NOISE_SWEEP],
        "uniform_baseline": float(uniform_baseline),
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A contract-shaped no-op head: zero logits -> uniform causal attention.

    Same signature as a real ``model_fn``. Used by the pipeline smoke test;
    pure NumPy, never crashes, returns correctly shaped output.
    """
    def _fn(residual: np.ndarray) -> np.ndarray:
        L = residual.shape[0]
        return np.zeros((L, L), dtype=np.float32)

    return _fn
