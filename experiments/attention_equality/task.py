"""Task for the `attention_equality` goal.

Question
--------
Does an attention head implement an *equality lookup*?  Given a query token,
a head that "computes equality" routes its attention mass onto earlier key
positions that hold the **same** token as the query.

Each sequence plants exactly one equal pair: a target token `t` appears at two
positions `p1 < p2`, and every other position holds a distinct distractor
token (`!= t`, and distinct from each other).  The only equal pair in the
sequence is therefore `(p1, p2)`.  We treat `p2` as the query of interest and
ask: how much attention mass does the head place on the single matching key
`p1`?

`match_mass = attn[p2, p1]` is the metric per sequence (bigger is better, in
`[0, 1]`).  A perfect equality head -> ~1.0; a uniform head -> ~1/(p2+1).

The difficulty axis is sequence length `L` (more positions = more distractors
= a harder lookup).  A uniform-attention baseline is computed analytically
under identical conditions, so beating it is what's meaningful.

Contract
--------
- `generate(seed=0, L=16) -> Batch`   deterministic for a given (seed, L).
- `evaluate(model_fn) -> dict`        runs the canonical sweep, returns payload.
- `random_model_fn() -> ModelFn`      returns a uniform-attention model_fn.

`model_fn` signature (the goal's contract with attempts):
    model_fn(batch: Batch) -> np.ndarray of shape (B, L, L), float, row-stochastic
    over the causally-allowed keys (mass 0 on disallowed positions).

Pure Python / NumPy.  No I/O, no network, no torch, no GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Fixed across the whole goal.  Vocab is kept comfortably larger than the
# longest swept L so distractors can always be made distinct.
V = 128                      # vocabulary size
B = 256                      # sequences per slice
CANONICAL_L = 16             # canonical measurement condition
L_SWEEP = [8, 16, 32, 64]    # difficulty axis

ModelFn = Callable[["Batch"], np.ndarray]


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray   # (B, L) int32
    mask: np.ndarray     # (B, L, L) bool, causal (lower-triangular incl. diagonal)
    p1: np.ndarray       # (B,) int — earlier position of the planted equal pair
    p2: np.ndarray       # (B,) int — later position (the query of interest)
    L: int               # sequence length
    V: int               # vocabulary size


def generate(seed: int = 0, L: int = CANONICAL_L) -> Batch:
    """Deterministic batch: same (seed, L) -> same batch.

    Each of the `B` sequences contains exactly one equal pair (p1, p2); all
    other positions hold distinct distractor tokens != the target token.
    """
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if L > V:
        raise ValueError(f"L={L} exceeds vocab V={V}; cannot keep distractors distinct")

    rng = np.random.default_rng(seed)

    tokens = np.zeros((B, L), dtype=np.int32)
    p1 = np.zeros(B, dtype=np.int64)
    p2 = np.zeros(B, dtype=np.int64)

    for b in range(B):
        # Pick the two equal-token positions p1 < p2 (p2 >= 1 so it has a past).
        pair = rng.choice(L, size=2, replace=False)
        a, c = int(min(pair)), int(max(pair))
        p1[b], p2[b] = a, c

        # Target token and distinct distractors, all drawn without replacement
        # so the (p1, p2) pair is the ONLY equal pair in the sequence.
        symbols = rng.choice(V, size=L, replace=False)  # L distinct symbols
        target = symbols[0]
        distractors = symbols[1:]  # length L-1, all != target and distinct

        di = 0
        for i in range(L):
            if i == a or i == c:
                tokens[b, i] = target
            else:
                tokens[b, i] = distractors[di]
                di += 1

    mask = np.broadcast_to(np.tril(np.ones((L, L), dtype=bool)), (B, L, L)).copy()
    return Batch(tokens=tokens, mask=mask, p1=p1, p2=p2, L=L, V=V)


def _validate_attn(attn: np.ndarray, batch: Batch) -> np.ndarray:
    """Coerce + shape-check the model output; return float64 (B, L, L)."""
    attn = np.asarray(attn, dtype=np.float64)
    expected = (B, batch.L, batch.L)
    if attn.shape != expected:
        raise ValueError(
            f"model_fn returned shape {attn.shape}, expected {expected}"
        )
    if not np.all(np.isfinite(attn)):
        raise ValueError("model_fn returned non-finite attention weights")
    return attn


def _slice_stats(attn: np.ndarray, batch: Batch) -> dict:
    """Compute per-slice scalars from one batch of attention weights."""
    attn = _validate_attn(attn, batch)

    # Row-stochasticity over allowed keys (deviation from 1.0).
    masked = attn * batch.mask
    rowsums = masked.sum(axis=2)  # (B, L)
    rowsum_max_dev = float(np.max(np.abs(rowsums - 1.0)))

    # match_mass = attention from query p2 onto the matching key p1.
    rows = np.arange(B)
    match_mass = attn[rows, batch.p2, batch.p1]  # (B,)
    mean_match_mass = float(np.mean(match_mass))

    # Analytic uniform-attention baseline under identical conditions:
    # a uniform row over the (p2+1) causally-allowed keys puts 1/(p2+1) on p1.
    uniform = 1.0 / (batch.p2.astype(np.float64) + 1.0)
    uniform_baseline = float(np.mean(uniform))

    return {
        "L": int(batch.L),
        "n_eval": int(B),
        "match_mass": mean_match_mass,
        "uniform_baseline": uniform_baseline,
        "attn_rowsum_max_dev": rowsum_max_dev,
    }


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical L-sweep and return the score payload."""
    sweep = []
    rowsum_devs = []
    for L in L_SWEEP:
        batch = generate(seed=0, L=L)
        attn = model_fn(batch)
        rec = _slice_stats(attn, batch)
        sweep.append(rec)
        rowsum_devs.append(rec["attn_rowsum_max_dev"])

    canonical = next(r for r in sweep if r["L"] == CANONICAL_L)

    return {
        "version": 1,
        "config": {
            "V": V,
            "B": B,
            "canonical_L": CANONICAL_L,
            "L_sweep": list(L_SWEEP),
            "causal": True,
        },
        "canonical": dict(canonical),
        "sweep": sweep,
        "attn_rowsum_max_dev": float(max(rowsum_devs)) if rowsum_devs else 0.0,
    }


def random_model_fn() -> ModelFn:
    """Return a uniform-attention model_fn (the no-mechanism reference).

    Pure NumPy.  Output is row-stochastic over the causally-allowed keys, so it
    exercises the full payload contract while scoring near the uniform baseline.
    """

    def _fn(batch: Batch) -> np.ndarray:
        mask = batch.mask.astype(np.float64)          # (B, L, L)
        counts = mask.sum(axis=2, keepdims=True)      # (B, L, 1), always >= 1
        return mask / counts

    return _fn
