"""Task for the attention_palindrome goal.

Pure Python + NumPy. No torch, no sklearn, no I/O, no network.

The question: can a model detect *palindromes* — a property that lives in the
*alignment* of mirrored positions, not in the bag of tokens? We probe this with
a difficulty sweep over how many mirrored pairs are broken in the negatives.
A genuine mirror-comparison mechanism (e.g. a head attending position i -> L-1-i
and checking token equality) stays sharp even when only a single pair is broken;
a histogram / shortcut readout collapses, because the token multiset is (by
construction) almost uninformative about palindrome-ness.

Exports
-------
generate(seed) -> Batch
evaluate(model_fn) -> payload   (dict consumed verbatim by benchmark.score)
random_model_fn() -> ModelFn    (shape-correct dummy for the smoke test)

The model_fn contract (the goal's interface with attempts):

    model_fn(batch: Batch) -> np.ndarray of shape (n_seq,), float

    Returns a real-valued *palindrome score* per sequence, higher = more
    palindrome-like. The absolute scale is irrelevant; only the ordering of
    scores within the batch is scored (the benchmark uses rank-based AUC).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ---- canonical measurement condition ----------------------------------------
SEQ_LEN = 16                     # sequence length (even)
HALF = SEQ_LEN // 2             # number of mirror pairs
VOCAB = 8                       # token alphabet size
N_POS = 256                     # number of positive (perfect-palindrome) seqs
N_NEG = 256                     # number of negatives per difficulty slice
MISMATCH_SWEEP = (1, 2, 4, 8)  # # of broken mirror pairs in the negatives
CANONICAL_SEED = 42


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray       # (n_seq, SEQ_LEN) int32
    is_palindrome: np.ndarray  # (n_seq,) bool  — True for the perfect palindromes
    mismatch: np.ndarray     # (n_seq,) int32 — 0 for positives, k for slice-k negatives


# model_fn: Batch -> (n_seq,) float scores; higher = more palindrome-like.
ModelFn = Callable[[Batch], np.ndarray]


# ---- data generation --------------------------------------------------------
def _make_palindrome(rng: np.random.Generator) -> np.ndarray:
    first = rng.integers(0, VOCAB, size=HALF, dtype=np.int32)
    return np.concatenate([first, first[::-1]])


def _break_pairs(pal: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Return a copy of `pal` with exactly k mirror pairs broken.

    A pair i is broken by reassigning the right-hand token (position L-1-i) to a
    value different from its mirror. Class-uninformative w.r.t. the token
    histogram: the perturbation direction does not correlate with the label, so a
    bag-of-tokens readout sits at chance."""
    seq = pal.copy()
    idx = rng.choice(HALF, size=k, replace=False)
    for i in idx:
        left = int(seq[i])
        # pick a different token uniformly from the remaining VOCAB-1 values
        new = int(rng.integers(0, VOCAB - 1))
        if new >= left:
            new += 1
        seq[SEQ_LEN - 1 - i] = new
    return seq


def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed. seed=0 maps to the canonical seed (42).

    Builds a shared pool of N_POS perfect palindromes plus, for each k in
    MISMATCH_SWEEP, N_NEG negatives differing from a palindrome in exactly k
    mirror pairs. All slices reuse the same positive pool when AUC is computed."""
    if seed == 0:
        seed = CANONICAL_SEED
    rng = np.random.default_rng(seed)

    tokens_list = []
    mismatch_list = []

    # positives
    for _ in range(N_POS):
        tokens_list.append(_make_palindrome(rng))
        mismatch_list.append(0)

    # negatives, one block per difficulty slice
    for k in MISMATCH_SWEEP:
        for _ in range(N_NEG):
            base = _make_palindrome(rng)
            tokens_list.append(_break_pairs(base, k, rng))
            mismatch_list.append(k)

    tokens = np.asarray(tokens_list, dtype=np.int32)
    mismatch = np.asarray(mismatch_list, dtype=np.int32)
    is_palindrome = mismatch == 0
    return Batch(tokens=tokens, is_palindrome=is_palindrome, mismatch=mismatch)


# ---- scoring helpers (pure NumPy) -------------------------------------------
def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties resolved by their mean rank — scipy-style."""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    sorter = np.argsort(a, kind="mergesort")
    a_sorted = a[sorter]
    obs = np.concatenate(([True], a_sorted[1:] != a_sorted[:-1]))
    dense = np.cumsum(obs)                      # dense rank per sorted element
    bounds = np.concatenate((np.nonzero(obs)[0], [n]))
    starts = bounds[:-1].astype(np.float64)
    ends = bounds[1:].astype(np.float64)
    avg = (starts + 1.0 + ends) / 2.0          # mean ordinal rank within each group
    ranks_sorted = avg[dense - 1]
    ranks = np.empty(n, dtype=np.float64)
    ranks[sorter] = ranks_sorted
    return ranks


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC = P(score(pos) > score(neg)), ties counted as 0.5.

    Undefined (empty group) -> 0.5 (chance), documented edge case."""
    pos = np.asarray(pos, dtype=np.float64).ravel()
    neg = np.asarray(neg, dtype=np.float64).ravel()
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        return 0.5
    combined = np.concatenate([pos, neg])
    ranks = _rankdata(combined)
    rank_pos_sum = float(ranks[:n_pos].sum())
    u = rank_pos_sum - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _histograms(tokens: np.ndarray) -> np.ndarray:
    n = tokens.shape[0]
    h = np.zeros((n, VOCAB), dtype=np.float64)
    for i in range(n):
        h[i] = np.bincount(tokens[i], minlength=VOCAB)
    return h


def _ridge_baseline_scores(batch: Batch) -> np.ndarray:
    """Best-possible *linear-on-token-histogram* readout, closed form (ridge).

    No fitting library: w = (XᵀX + λI)⁻¹ Xᵀ(y - ȳ), intercept folds out via
    centering. This is the no-mechanism reference — palindrome-ness is essentially
    orthogonal to the token multiset, so this should sit near AUC 0.5."""
    X = _histograms(batch.tokens)
    y = batch.is_palindrome.astype(np.float64)
    xbar = X.mean(axis=0, keepdims=True)
    ybar = float(y.mean())
    Xc = X - xbar
    yc = y - ybar
    lam = 1.0
    A = Xc.T @ Xc + lam * np.eye(VOCAB)
    w = np.linalg.solve(A, Xc.T @ yc)
    return (Xc @ w) + ybar


def _sweep_records(scores: np.ndarray, batch: Batch) -> list[dict]:
    pos_scores = scores[batch.is_palindrome]
    records = []
    for k in MISMATCH_SWEEP:
        neg_scores = scores[batch.mismatch == k]
        records.append({
            "mismatch": int(k),
            "auc": _auc(pos_scores, neg_scores),
            "n_pos": int(pos_scores.size),
            "n_neg": int(neg_scores.size),
        })
    return records


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn on the canonical batch and return the benchmark payload."""
    batch = generate(CANONICAL_SEED)
    scores = model_fn(batch)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n_seq = batch.tokens.shape[0]
    if scores.shape != (n_seq,):
        raise ValueError(
            f"model_fn returned scores of shape {scores.shape}, expected {(n_seq,)}"
        )
    if not np.all(np.isfinite(scores)):
        raise ValueError("model_fn returned non-finite scores")

    baseline_scores = _ridge_baseline_scores(batch)

    return {
        "version": 1,
        "canonical_seed": CANONICAL_SEED,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB,
        "n_pos": int(N_POS),
        "n_neg_per_slice": int(N_NEG),
        "mismatch_sweep": list(MISMATCH_SWEEP),
        "sweep": _sweep_records(scores, batch),
        "linear_baseline": _sweep_records(baseline_scores, batch),
    }


def random_model_fn() -> ModelFn:
    """Shape-correct dummy model_fn for the smoke test: deterministic random
    scores of the right shape. Pure NumPy, no torch, no GPU. Expected AUC ~0.5."""
    def _fn(batch: Batch) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.standard_normal(batch.tokens.shape[0]).astype(np.float64)
    return _fn
