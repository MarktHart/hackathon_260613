"""
Synthetic attention-deduplication task.

The question: does an attention mechanism route a *duplicate* token's query
back to the position of its **previous occurrence**? This is the behaviour of a
"duplicate-token head" — a well-known mechanistic-interpretability motif and a
precursor to induction. We measure how much attention mass a query at a
repeated token places on the most recent earlier position holding the same
token, and how often that earlier position is the arg-max key.

Exports:
    generate(seed=0) -> Batch
    evaluate(model_fn) -> payload dict   (shape consumed by benchmark.score)
    random_model_fn() -> ModelFn         (smoke-test / reference stub)

Pure Python + NumPy. No torch, no GPU, no I/O.
"""

from dataclasses import dataclass, field
from typing import Callable, Tuple
import numpy as np

# ──────────────────────────────────────────────────────────────────────
# Canonical, fixed configuration
# ──────────────────────────────────────────────────────────────────────
_SEQ_LEN = 24
_N_SEQS = 64
_VOCAB_SIZE = 64
_DUP_RATES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7)
_CANONICAL_DUP_RATE = 0.5


# ──────────────────────────────────────────────────────────────────────
# Batch
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Batch:
    """
    A sweep over duplicate-token density. Each slice is a dict:
        {
            "dup_rate": float,
            "tokens":   np.ndarray[int]  (N, L)   token ids,
            "prev":     np.ndarray[int]  (N, L)   previous-occurrence index of
                                                  each token, or -1 if first-seen.
        }
    """
    slices: Tuple[dict, ...]
    seq_len: int = _SEQ_LEN
    n_seqs: int = _N_SEQS
    vocab_size: int = _VOCAB_SIZE
    dup_rates: Tuple[float, ...] = _DUP_RATES
    canonical_dup_rate: float = _CANONICAL_DUP_RATE
    seed: int = 0


def _make_slice(rng: np.random.Generator, p: float) -> dict:
    """One slice of N sequences with duplicate density ~p, plus prev-occurrence."""
    N, L, V = _N_SEQS, _SEQ_LEN, _VOCAB_SIZE
    tokens = np.zeros((N, L), dtype=np.int64)
    prev = np.full((N, L), -1, dtype=np.int64)

    for s in range(N):
        last_occ: dict[int, int] = {}      # token -> most recent index seen
        for i in range(L):
            make_dup = (i > 0) and (len(last_occ) > 0) and (rng.random() < p)
            if make_dup:
                # copy the token at a random earlier position -> guaranteed dup
                j = int(rng.integers(0, i))
                tok = int(tokens[s, j])
            else:
                # prefer a token never seen in this sequence (a true first-seen)
                unseen = [t for t in range(V) if t not in last_occ]
                if unseen:
                    tok = int(unseen[int(rng.integers(0, len(unseen)))])
                else:
                    tok = int(rng.integers(0, V))
            prev[s, i] = last_occ.get(tok, -1)     # previous occurrence (or -1)
            tokens[s, i] = tok
            last_occ[tok] = i

    return {"dup_rate": float(p), "tokens": tokens, "prev": prev}


def generate(seed: int = 0) -> Batch:
    """
    Deterministic generator: same seed -> identical Batch. The canonical
    condition is seed=0; non-zero seeds reshuffle the token streams but keep the
    same sweep axis and shapes.
    """
    rng = np.random.default_rng(int(seed))
    slices = tuple(_make_slice(rng, p) for p in _DUP_RATES)
    return Batch(slices=slices, seed=int(seed))


# ──────────────────────────────────────────────────────────────────────
# Model function contract
# ──────────────────────────────────────────────────────────────────────
# ModelFn signature:
#     model_fn(tokens: np.ndarray[int, (N, L)]) -> np.ndarray[float, (N, L, L)]
# Returns causal attention weights; attn[s, q, k] is the weight from query q to
# key k. Rows should be row-stochastic over causal keys (k <= q). The evaluator
# re-normalises defensively, so non-stochastic output is tolerated but the
# mechanism the metric rewards is "put mass on the previous occurrence".
ModelFn = Callable[[np.ndarray], np.ndarray]


def random_model_fn() -> ModelFn:
    """
    Reference stub: random causal row-stochastic attention. Pure NumPy. Used by
    the pipeline smoke test and as the no-mechanism sanity floor (it should not
    beat the uniform-causal baseline meaningfully).
    """
    rng = np.random.default_rng(0)

    def _fn(tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens)
        n, L = tokens.shape
        causal = np.tril(np.ones((L, L), dtype=np.float64))
        w = (rng.random((n, L, L)) + 1e-6) * causal[None]
        w = w / w.sum(axis=-1, keepdims=True)
        return w

    return _fn


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────
def _causal_normalize(attn: np.ndarray, L: int) -> np.ndarray:
    """Zero non-causal entries and renormalise each row; empty rows -> uniform."""
    causal = np.tril(np.ones((L, L), dtype=np.float64))
    a = np.where(causal[None] > 0, attn, 0.0)
    s = a.sum(axis=-1, keepdims=True)                  # (N, L, 1)
    uniform = causal / causal.sum(axis=-1, keepdims=True)   # (L, L)
    a = np.where(s > 1e-12, a / np.maximum(s, 1e-12), uniform[None])
    return a


def _mean(vals) -> float:
    return float(np.mean(vals)) if len(vals) else 0.0


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run `model_fn` over the canonical batch (seed=0) and return the payload dict
    exactly as benchmark.score expects it. Attempts never build this dict — they
    hand over a model_fn and receive a ready-to-record payload.
    """
    batch = generate(0)
    sweep = []

    for sl in batch.slices:
        p = sl["dup_rate"]
        tokens = sl["tokens"]
        prev = sl["prev"]
        n, L = tokens.shape

        attn = np.asarray(model_fn(tokens), dtype=np.float64)
        if attn.shape != (n, L, L):
            raise ValueError(
                f"model_fn must return (N, L, L) = {(n, L, L)}, got {attn.shape}"
            )
        a = _causal_normalize(attn, L)

        dup_idx = np.argwhere(prev >= 0)        # (n_dup, 2) -> (s, i)
        fs_idx = np.argwhere(prev < 0)          # first-seen positions

        dup_mass, dup_hit, base_mass = [], [], []
        for s, i in dup_idx:
            tgt = int(prev[s, i])
            row = a[s, i]
            dup_mass.append(float(row[tgt]))
            dup_hit.append(1.0 if int(np.argmax(row)) == tgt else 0.0)
            base_mass.append(1.0 / (i + 1))     # uniform-causal mass on the target

        fs_self = [float(a[s, i, i]) for s, i in fs_idx]

        sweep.append({
            "dup_rate": float(p),
            "n_dup_positions": int(len(dup_idx)),
            "n_first_seen": int(len(fs_idx)),
            "dedup_mass": _mean(dup_mass),
            "dedup_accuracy": _mean(dup_hit),
            "first_seen_self_mass": _mean(fs_self),
            "baseline_dedup_mass": _mean(base_mass),
            "baseline_dedup_accuracy": _mean(base_mass),
        })

    payload = {
        "version": 1,
        "task": "attention_dedupe",
        "seed": 0,
        "seq_len": _SEQ_LEN,
        "vocab_size": _VOCAB_SIZE,
        "n_seqs": _N_SEQS,
        "dup_rates": list(_DUP_RATES),
        "canonical_dup_rate": _CANONICAL_DUP_RATE,
        "sweep": sweep,
    }
    return payload
