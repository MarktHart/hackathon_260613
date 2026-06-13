"""Data generator and evaluator for the attention_shift_by_k goal.

Exports:
    generate(seed=0) -> Batch
    evaluate(model_fn) -> payload dict        (matches benchmark.score contract)
    random_model_fn() -> ModelFn              (uniform-attention no-signal baseline)

Pure NumPy. No torch, no GPU, no I/O, no network.

The model function contract (see README.md):

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        # input_ids: int32, shape (batch, seq_len)
        # returns:   float, shape (batch, n_heads, seq_len, seq_len)
        #            row-stochastic over the last (key) axis.

The goal asks: does *some* attention head implement a fixed "shift by k"
pattern, i.e. query position ``i`` attends to key position ``i - k``? We sweep
``k`` over several offsets and, for each, report the single best head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np

# ---- Canonical constants — must match README.md ----
CANONICAL_SEQ_LEN = 32
CANONICAL_BATCH_SIZE = 8
CANONICAL_VOCAB_SIZE = 64
CANONICAL_SEED = 0
K_VALUES = (1, 2, 3, 4, 8)   # shift offsets swept; all < SEQ_LEN
CANONICAL_K = 1              # headline measurement condition (previous token)

ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray       # shape (batch, seq_len), int32
    seq_len: int
    batch_size: int
    vocab_size: int
    k_values: tuple[int, ...]


def generate(seed: int = CANONICAL_SEED) -> Batch:
    """Deterministic random integer sequences.

    Same seed -> same batch. The seed is used (not ignored), so different seeds
    give different sequences for optional robustness checks. The canonical
    measurement condition uses ``seed = 0``. Token identities are irrelevant to
    the position-based shift pattern, but real models consume tokens, so we
    hand over genuine integer ids.
    """
    rng = np.random.default_rng(seed)
    input_ids = rng.integers(
        low=0,
        high=CANONICAL_VOCAB_SIZE,
        size=(CANONICAL_BATCH_SIZE, CANONICAL_SEQ_LEN),
        dtype=np.int32,
    )
    return Batch(
        input_ids=input_ids,
        seq_len=CANONICAL_SEQ_LEN,
        batch_size=CANONICAL_BATCH_SIZE,
        vocab_size=CANONICAL_VOCAB_SIZE,
        k_values=K_VALUES,
    )


def random_model_fn() -> ModelFn:
    """Return a model_fn emitting uniform attention over all key positions.

    A valid no-signal baseline with exactly the real model_fn signature and
    output shape. Pure NumPy. Mass on any single target key is ``1 / seq_len``.
    """
    n_heads = 4  # arbitrary but fixed; evaluate() reads H from the output shape

    def _fn(input_ids: np.ndarray) -> np.ndarray:
        batch, seq_len = input_ids.shape
        return np.full(
            (batch, n_heads, seq_len, seq_len),
            1.0 / seq_len,
            dtype=np.float32,
        )

    return _fn


def evaluate(model_fn: ModelFn) -> dict:
    """Run ``model_fn`` on the canonical batch and assemble the payload.

    For each offset ``k`` in ``K_VALUES`` and each head, measure the mean
    attention mass placed by query ``i`` on key ``i - k`` (over valid queries
    ``i = k .. L-1`` and over the batch). The best head per ``k`` is the one
    with the highest such mass. Returns the payload dict that
    ``benchmark.score`` consumes.
    """
    batch = generate(CANONICAL_SEED)
    attn = np.asarray(model_fn(batch.input_ids))

    if attn.ndim != 4:
        raise ValueError(
            f"model_fn must return a 4D array (batch, n_heads, seq_len, seq_len), "
            f"got shape {attn.shape}"
        )
    B, H, L_q, L_k = attn.shape
    if B != batch.batch_size or L_q != batch.seq_len or L_k != batch.seq_len:
        raise ValueError(
            f"model_fn returned shape {attn.shape}, expected "
            f"({batch.batch_size}, n_heads, {batch.seq_len}, {batch.seq_len})"
        )
    if H <= 0:
        raise ValueError("model_fn must return at least one head")

    attn = attn.astype(np.float64, copy=False)
    if not np.all(np.isfinite(attn)):
        raise ValueError("model_fn returned non-finite attention weights")
    if np.any(attn < -1e-6):
        raise ValueError("attention weights must be non-negative")

    # Renormalise per query so each row sums to 1 over keys (defensive).
    row_sums = attn.sum(axis=-1, keepdims=True)
    attn = np.where(row_sums > 0, attn / row_sums, 0.0)

    uniform_baseline = 1.0 / batch.seq_len

    sweep_records = []
    for k in batch.k_values:
        qi = np.arange(k, batch.seq_len)        # valid query positions
        kj = qi - k                              # shift-by-k target keys

        # Mass on the correct target for each (batch, head, valid query).
        correct = attn[:, :, qi, kj]             # (B, H, L-k)
        per_head_mass = correct.mean(axis=(0, 2))  # (H,)

        # Argmax accuracy: how often the head's peak key IS the target.
        peak = attn[:, :, qi, :].argmax(axis=-1)   # (B, H, L-k)
        hit = (peak == kj[None, None, :])           # broadcast over B, H
        per_head_argmax = hit.mean(axis=(0, 2))     # (H,)

        best_idx = int(np.argmax(per_head_mass))
        sweep_records.append({
            "k": int(k),
            "best_head_index": best_idx,
            "best_head_mass": float(per_head_mass[best_idx]),
            "best_head_argmax_acc": float(per_head_argmax[best_idx]),
            "mean_head_mass": float(per_head_mass.mean()),
            "uniform_baseline": float(uniform_baseline),
        })

    return {
        "version": 1,                 # must match benchmark.VERSION
        "model_name": "unknown",      # attempt's main.py may overwrite
        "seq_len": batch.seq_len,
        "batch_size": batch.batch_size,
        "vocab_size": batch.vocab_size,
        "num_heads": int(H),
        "k_values": [int(k) for k in batch.k_values],
        "canonical_k": int(CANONICAL_K),
        "uniform_baseline": float(uniform_baseline),
        "sweep": sweep_records,
    }
