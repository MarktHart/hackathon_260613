"""
Task for `attention_distance_compare`.

Question: how strongly does a model's attention concentrate on *nearby* key
positions versus distant ones, and do individual layers/heads differ in their
distance preference? We bin every query->key attention weight by the positional
distance |i - j|, average the weight per bin across the whole batch (globally,
and per layer/head), and hand the resulting per-bin curves to the benchmark,
which reduces them to a distance-decay slope, a locality fraction, and an
entropy.

The model contract (`model_fn`) is intentionally narrow:

    model_fn(input_ids: np.ndarray[int32, (B, S)]) -> {"attention": np.ndarray}

where "attention" is a per-query-normalised attention tensor of shape either
    (n_layers, n_heads, B, S, S)        # batched
or  (n_layers, n_heads, S, S)           # single-pattern, broadcast over B
with each row over the final (key) axis summing to 1.

Pure NumPy; no I/O, no network, no torch, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

ModelFn = Callable[[np.ndarray], dict]


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray  # shape (BATCH_SIZE, SEQ_LEN), dtype=int32


# --- Canonical configuration (the measurement condition every attempt uses) ---
SEQ_LEN = 64
BATCH_SIZE = 32
N_LAYERS = 4
N_HEADS = 8
VOCAB_SIZE = 1000
CANONICAL_SEED = 0

# Half-open distance bin edges: [0,1), [1,2), ..., [33,64).  Distance |i-j|
# ranges over 0..SEQ_LEN-1.
DISTANCE_BIN_EDGES = np.array(
    [0, 1, 2, 3, 4, 5, 7, 11, 17, 33, 64], dtype=np.int64
)
# Representative center per bin (used as the x-axis for the decay fit).
DISTANCE_BIN_CENTERS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 9.0, 13.5, 24.5, 48.0]
N_BINS = len(DISTANCE_BIN_CENTERS)

assert len(DISTANCE_BIN_EDGES) - 1 == N_BINS


def generate(seed: int = 0) -> Batch:
    """
    Deterministic random token sequences.

    The canonical measurement condition fixes the seed to ``CANONICAL_SEED``;
    the ``seed`` argument is accepted for API compatibility but ignored so that
    two attempts always score against identical data.
    """
    rng = np.random.default_rng(CANONICAL_SEED)
    input_ids = rng.integers(
        0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN), dtype=np.int32
    )
    return Batch(input_ids=input_ids)


def _bin_index_matrix(seq_len: int) -> np.ndarray:
    """Bin index for every (query, key) cell, shape (seq_len, seq_len)."""
    pos = np.arange(seq_len)
    dist = np.abs(pos[:, None] - pos[None, :])
    idx = np.searchsorted(DISTANCE_BIN_EDGES, dist, side="right") - 1
    return np.clip(idx, 0, N_BINS - 1)


def _mean_attn_per_bin(attn_bss: np.ndarray, bin_idx: np.ndarray) -> np.ndarray:
    """
    Mean attention weight per distance bin for one (layer, head).

    Args:
        attn_bss: (B, S, S) per-query-normalised attention.
        bin_idx:  (S, S) bin index per cell.
    Returns:
        (N_BINS,) float64 mean attention weight of the cells falling in each
        bin (averaged over batch and query position).  Empty bins -> 0.0.
    """
    batch = attn_bss.shape[0]
    flat_attn = attn_bss.reshape(batch, -1)            # (B, S*S)
    flat_bins = bin_idx.reshape(-1)                     # (S*S,)
    out = np.zeros(N_BINS, dtype=np.float64)
    for b in range(N_BINS):
        mask = flat_bins == b
        if np.any(mask):
            out[b] = float(flat_attn[:, mask].mean())
    return out


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run ``model_fn`` over the canonical batch and build the benchmark payload.
    """
    batch = generate()
    out = model_fn(batch.input_ids)

    if not isinstance(out, dict) or "attention" not in out:
        raise ValueError("model_fn must return a dict with key 'attention'")

    attention = np.asarray(out["attention"], dtype=np.float64)

    if attention.ndim == 4:
        # (L, H, S, S) -> broadcast over the batch axis.
        attention = np.repeat(
            attention[:, :, None, :, :], BATCH_SIZE, axis=2
        )
    elif attention.ndim != 5:
        raise ValueError(
            f"attention must be 4D (L,H,S,S) or 5D (L,H,B,S,S), "
            f"got {attention.ndim}D"
        )

    expected = (N_LAYERS, N_HEADS, BATCH_SIZE, SEQ_LEN, SEQ_LEN)
    if attention.shape != expected:
        raise ValueError(
            f"attention shape {attention.shape} != expected {expected}"
        )

    # Per-query normalisation check (rows over the key axis sum to 1).
    row_sums = attention.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        max_err = float(np.max(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"attention not per-query-normalised: max|rowsum-1| = {max_err:.4f}"
        )

    bin_idx = _bin_index_matrix(SEQ_LEN)

    per_lh: list = []  # (L, H, N_BINS)
    for l in range(N_LAYERS):
        layer_rows = []
        for h in range(N_HEADS):
            layer_rows.append(
                _mean_attn_per_bin(attention[l, h], bin_idx).tolist()
            )
        per_lh.append(layer_rows)

    per_lh_arr = np.asarray(per_lh, dtype=np.float64)        # (L, H, N_BINS)
    mean_per_bin = per_lh_arr.mean(axis=(0, 1)).tolist()     # (N_BINS,)

    # Uniform-attention baseline: every cell = 1/SEQ_LEN, so the per-bin mean is
    # 1/SEQ_LEN in every bin regardless of distance (a flat curve, slope ~ 0).
    uniform_per_bin = [1.0 / SEQ_LEN] * N_BINS

    payload = {
        "version": 1,
        "canonical_config": {
            "seq_len": SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "vocab_size": VOCAB_SIZE,
            "seed": CANONICAL_SEED,
            "distance_bin_edges": DISTANCE_BIN_EDGES.tolist(),
        },
        "distance_bins": list(DISTANCE_BIN_CENTERS),
        "mean_attn_per_bin": mean_per_bin,
        "uniform_baseline_per_bin": uniform_per_bin,
        "mean_attn_per_layer_head_bin": per_lh,
    }
    return payload


def random_model_fn() -> ModelFn:
    """
    A ``model_fn`` with the real signature whose attention is uniform over keys
    (zero distance structure).  Pure NumPy; used by the pipeline smoke test.
    """

    def _fn(input_ids: np.ndarray) -> dict:
        batch, seq_len = np.asarray(input_ids).shape
        attn = np.full(
            (N_LAYERS, N_HEADS, batch, seq_len, seq_len),
            1.0 / seq_len,
            dtype=np.float64,
        )
        return {"attention": attn}

    return _fn
