"""Data generator and evaluator for the attention_not goal.

Exports:
    generate(seed=0) -> Batch
    evaluate(model_fn) -> payload dict   (consumed verbatim by benchmark.score)
    random_model_fn() -> ModelFn

The goal asks whether a single attention head can implement a logical NOT /
inhibitory gate: attend from a query position to the A-token *only when* an
inhibitory feature B is absent, and suppress that attention when B is present.
We measure how cleanly this NOT separates B=0 from B=1, and how robust that
separation is as the "attend" and "suppress" feature directions are forced
into superposition (cosine theta > 0).

The model_fn contract (the goal's interface with attempts):

    model_fn(batch: Batch) -> dict
        returns {"attn_weights": np.ndarray of shape
                 (n_seq, seq_len, seq_len), rows summing to ~1 over the last
                 axis}.  attn_weights[:, QUERY_POS, A_POS] is the attention
                 mass the query position places on the A-token.

Pure NumPy. Deterministic for a given seed. No I/O, no network, no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Fixed sequence layout: [A-token, B-token, query, answer].
A_POS = 0
B_POS = 1
QUERY_POS = 2
ANS_POS = 3
SEQ_LEN = 4

# Canonical superposition sweep over cos(theta) between the attend/suppress
# feature directions. Index 0 (orthogonal) is the canonical anchor.
CANONICAL_COS = (0.0, 0.2, 0.4, 0.6, 0.8)

D_MODEL = 64
D_HEAD = 16
N_SEQ = 600


@dataclass(frozen=True)
class Batch:
    """One superposition condition.

    Fully determined by (cos_theta, seed). `tokens` carries the fixed layout;
    `feat_A` / `feat_B` are the per-sequence binary features; the projection
    matrices give attempts a concrete head geometry to work from.
    """

    cos_theta: float
    tokens: np.ndarray          # (n_seq, SEQ_LEN) int32 token ids
    feat_A: np.ndarray          # (n_seq,) int32 in {0,1}
    feat_B: np.ndarray          # (n_seq,) int32 in {0,1}
    e_A: np.ndarray             # (d_model,) attend-feature direction (unit)
    e_B: np.ndarray             # (d_model,) suppress-feature direction (unit)
    W_Q: np.ndarray             # (d_model, d_head)
    W_K: np.ndarray             # (d_model, d_head)
    W_V: np.ndarray             # (d_model, d_model)
    W_O: np.ndarray             # (d_model, d_model)


ModelFn = Callable[[Batch], dict]


def _directions(d_model: int, cos_theta: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors e_A, e_B with e_A . e_B == cos_theta."""
    e_A = rng.normal(size=d_model)
    e_A /= np.linalg.norm(e_A)
    o = rng.normal(size=d_model)
    o -= (o @ e_A) * e_A
    o /= np.linalg.norm(o)
    sin = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    e_B = cos_theta * e_A + sin * o
    return e_A, e_B


def generate(seed: int = 0, cos_theta: float = 0.0) -> Batch:
    """Deterministic batch for one superposition condition.

    Same (seed, cos_theta) -> identical batch. `cos_theta` selects the angle
    between the attend (e_A) and suppress (e_B) feature directions.
    """
    rng = np.random.default_rng(seed)

    feat_A = rng.integers(0, 2, size=N_SEQ).astype(np.int32)
    feat_B = rng.integers(0, 2, size=N_SEQ).astype(np.int32)

    tokens = np.zeros((N_SEQ, SEQ_LEN), dtype=np.int32)
    tokens[:, A_POS] = 1
    tokens[:, B_POS] = 2
    tokens[:, QUERY_POS] = 3
    tokens[:, ANS_POS] = 4

    e_A, e_B = _directions(D_MODEL, cos_theta, rng)

    W_Q = rng.normal(size=(D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)
    W_K = rng.normal(size=(D_MODEL, D_HEAD)) / np.sqrt(D_MODEL)
    W_V = np.eye(D_MODEL, dtype=np.float64)
    W_O = np.eye(D_MODEL, dtype=np.float64)

    return Batch(
        cos_theta=float(cos_theta),
        tokens=tokens,
        feat_A=feat_A,
        feat_B=feat_B,
        e_A=e_A,
        e_B=e_B,
        W_Q=W_Q,
        W_K=W_K,
        W_V=W_V,
        W_O=W_O,
    )


def random_model_fn() -> ModelFn:
    """A model_fn with the real signature whose body returns zero-valued
    attention of the correct shape. Pure NumPy; used by the smoke test."""

    def _fn(batch: Batch) -> dict:
        n_seq = batch.tokens.shape[0]
        seq_len = batch.tokens.shape[1]
        return {
            "attn_weights": np.zeros((n_seq, seq_len, seq_len), dtype=np.float64),
        }

    return _fn


def _linear_baseline_fn() -> ModelFn:
    """No-NOT reference head: attends to the A-token whenever feature A is
    present, *ignoring* the inhibitory feature B. It therefore cannot
    distinguish B=0 from B=1 and should score ~chance on NOT-sharpness."""

    def _fn(batch: Batch) -> dict:
        n_seq, seq_len = batch.tokens.shape
        attn = np.zeros((n_seq, seq_len, seq_len), dtype=np.float64)
        # Query attends fully to A-token iff feat_A, else uniform over keys.
        for i in range(n_seq):
            if batch.feat_A[i] == 1:
                attn[i, QUERY_POS, A_POS] = 1.0
            else:
                attn[i, QUERY_POS, :] = 1.0 / seq_len
        return {"attn_weights": attn}

    return _fn


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random pos sample exceeds a random neg sample, with ties
    counted as 0.5. Returns 0.5 when either side is empty."""
    if pos.size == 0 or neg.size == 0:
        return 0.5
    gt = np.mean(pos[:, None] > neg[None, :])
    eq = np.mean(pos[:, None] == neg[None, :])
    return float(gt + 0.5 * eq)


def _condition_metrics(batch: Batch, model_out: dict) -> dict:
    """Reduce one condition's attention to scalar NOT metrics."""
    if not isinstance(model_out, dict) or "attn_weights" not in model_out:
        raise KeyError("model_fn must return a dict with key 'attn_weights'")
    aw = np.asarray(model_out["attn_weights"], dtype=np.float64)
    n_seq, seq_len = batch.tokens.shape
    if aw.shape != (n_seq, seq_len, seq_len):
        raise ValueError(
            f"attn_weights shape {aw.shape} != expected {(n_seq, seq_len, seq_len)}"
        )

    attn_to_A = aw[:, QUERY_POS, A_POS]  # (n_seq,)

    a1 = batch.feat_A == 1
    pos = attn_to_A[a1 & (batch.feat_B == 0)]  # NOT-active: should attend
    neg = attn_to_A[a1 & (batch.feat_B == 1)]  # inhibited: should not attend

    not_sharpness = _auc(pos, neg)
    suppression_gap = float(np.mean(pos) - np.mean(neg)) if (pos.size and neg.size) else 0.0

    # False-attend: query putting mass on A when feature A is absent. Lower is
    # better; reported as a specificity (1 - rate) to keep bigger-is-better.
    a0 = attn_to_A[batch.feat_A == 0]
    false_attend = float(np.mean(a0)) if a0.size else 0.0
    attend_specificity = float(1.0 - min(1.0, max(0.0, false_attend)))

    return {
        "cos": float(batch.cos_theta),
        "not_sharpness": float(not_sharpness),
        "suppression_gap": float(suppression_gap),
        "attend_specificity": attend_specificity,
    }


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical cos(theta) sweep and return the
    payload consumed verbatim by benchmark.score."""
    baseline_fn = _linear_baseline_fn()

    sweep: list[dict] = []
    baseline: list[dict] = []
    for i, cos in enumerate(CANONICAL_COS):
        batch = generate(seed=1234 + i, cos_theta=cos)
        sweep.append(_condition_metrics(batch, model_fn(batch)))
        baseline.append(_condition_metrics(batch, baseline_fn(batch)))

    return {
        "version": 1,
        "config": {
            "d_model": D_MODEL,
            "d_head": D_HEAD,
            "n_seq": N_SEQ,
            "seq_len": SEQ_LEN,
            "canonical_cos": list(CANONICAL_COS),
            "canonical_anchor": CANONICAL_COS[0],
        },
        "sweep": sweep,
        "baseline": baseline,
    }
