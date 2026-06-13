"""Synthetic bracket-matching task for the attention_brackets goal.

Pure Python / NumPy. No I/O, no network, no torch, no GPU.

The question: does an attention head route a *closing* bracket's query to its
*matching opening* bracket (the one the parser's stack would pop), rather than
to some positional heuristic such as "the nearest opener" or "the most recent
token"?

A `model_fn` here is a small, narrow callable:

    model_fn(tokens: np.ndarray[int32, (L,)]) -> np.ndarray[float, (L, L)]

It receives the integer token ids of ONE sequence and returns a causal,
row-stochastic attention matrix `A` where `A[q, k]` is the weight query
position `q` places on key position `k`. Rows must sum to ~1 (we renormalise
defensively). Any attempt — a trained transformer head, a hand-built circuit,
or a heuristic — can produce this shape.

`evaluate` sweeps over the maximum nesting depth of the generated sequences and
returns a payload that `benchmark.score` consumes. See README.md for the
contract.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Token ids.
OPEN = 0
CLOSE = 1
PAD = 2

# Canonical measurement condition.
DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 5)
CANONICAL_DEPTH: int = 3
N_PER_DEPTH: int = 64
SEQ_LEN: int = 24  # must be even
SEED_OFFSET: int = 0x5EED  # fixed offset for determinism

# A model_fn maps one sequence's tokens to a (L, L) attention matrix.
ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    """One deterministic batch of balanced-bracket sequences.

    `sequences[d]` is a tuple of token-id tuples (length SEQ_LEN each) whose
    maximum nesting depth is bounded by `d`. `matches[d]` is the parallel
    structure: for every closing-bracket position the index of its true
    matching opener; -1 everywhere else.
    """

    depths: tuple[int, ...]
    sequences: dict[int, tuple[tuple[int, ...], ...]]
    matches: dict[int, tuple[tuple[int, ...], ...]]
    seq_len: int
    canonical_depth: int
    n_per_depth: int


def _gen_sequence(rng: np.random.Generator, depth: int, length: int) -> list[int]:
    """Generate one balanced bracket sequence of exactly `length` tokens whose
    nesting never exceeds `depth`. Always valid and balanced (ends at depth 0)."""
    seq: list[int] = []
    h = 0  # current open depth
    for i in range(length):
        remaining = length - i  # tokens left to emit, including this one
        # We may open only if it stays within `depth` AND we can still close
        # everything in the tokens that remain after this one.
        can_open = (h < depth) and ((remaining - 1) >= (h + 1))
        can_close = h > 0
        if not can_close:
            seq.append(OPEN)
            h += 1
        elif not can_open:
            seq.append(CLOSE)
            h -= 1
        else:
            if rng.random() < 0.5:
                seq.append(OPEN)
                h += 1
            else:
                seq.append(CLOSE)
                h -= 1
    return seq


def _matching(seq: list[int]) -> list[int]:
    """For each position return the index of the matching opener if the token is
    a closer, else -1. Uses a stack, exactly like a parser."""
    match = [-1] * len(seq)
    stack: list[int] = []
    for i, tok in enumerate(seq):
        if tok == OPEN:
            stack.append(i)
        elif tok == CLOSE and stack:
            j = stack.pop()
            match[i] = j
    return match


def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed: same seed -> identical batch."""
    rng = np.random.default_rng(seed + SEED_OFFSET)
    sequences: dict[int, tuple[tuple[int, ...], ...]] = {}
    matches: dict[int, tuple[tuple[int, ...], ...]] = {}
    for d in DEPTHS:
        seqs = []
        ms = []
        for _ in range(N_PER_DEPTH):
            s = _gen_sequence(rng, d, SEQ_LEN)
            seqs.append(tuple(s))
            ms.append(tuple(_matching(s)))
        sequences[d] = tuple(seqs)
        matches[d] = tuple(ms)
    return Batch(
        depths=DEPTHS,
        sequences=sequences,
        matches=matches,
        seq_len=SEQ_LEN,
        canonical_depth=CANONICAL_DEPTH,
        n_per_depth=N_PER_DEPTH,
    )


def _normalise_causal(attn: np.ndarray, length: int) -> np.ndarray:
    """Coerce a model attention matrix into a causal, row-stochastic (L, L)."""
    a = np.asarray(attn, dtype=np.float64)
    if a.shape != (length, length):
        raise ValueError(f"model_fn returned shape {a.shape}, expected {(length, length)}")
    if not np.all(np.isfinite(a)):
        raise ValueError("model_fn returned non-finite attention weights")
    a = np.clip(a, 0.0, None)
    # Causal mask: a query at position q may only attend to keys k <= q.
    mask = np.tril(np.ones((length, length)))
    a = a * mask
    row = a.sum(axis=1, keepdims=True)
    row[row == 0.0] = 1.0  # avoid 0/0; leaves all-zero rows as zeros
    return a / row


def _evaluate_depth(model_fn: ModelFn, seqs, ms) -> dict:
    """Aggregate matching statistics over all sequences at one depth."""
    n_closers = 0
    acc_hits = 0.0
    mass_sum = 0.0
    baseline_sum = 0.0
    for seq_t, match_t in zip(seqs, ms):
        tokens = np.asarray(seq_t, dtype=np.int32)
        length = tokens.shape[0]
        attn = _normalise_causal(model_fn(tokens), length)
        for q, m in enumerate(match_t):
            if m < 0:
                continue
            n_closers += 1
            row = attn[q]
            # Attention mass placed on the true matching opener.
            mass_sum += float(row[m])
            # argmax routing hit (ties broken toward the lowest index).
            if int(np.argmax(row)) == m:
                acc_hits += 1.0
            # Uniform causal baseline: q may attend to keys 0..q (q+1 keys).
            baseline_sum += 1.0 / (q + 1)
    if n_closers == 0:
        return {
            "depth": 0,
            "n_closers": 0,
            "match_accuracy": 0.0,
            "match_mass": 0.0,
            "uniform_baseline_mass": 0.0,
        }
    return {
        "n_closers": n_closers,
        "match_accuracy": acc_hits / n_closers,
        "match_mass": mass_sum / n_closers,
        "uniform_baseline_mass": baseline_sum / n_closers,
    }


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload."""
    batch = generate(seed=0)
    sweep = []
    for d in batch.depths:
        rec = _evaluate_depth(model_fn, batch.sequences[d], batch.matches[d])
        rec["depth"] = d
        sweep.append(rec)
    return {
        "version": 1,
        "config": {
            "depths": list(batch.depths),
            "canonical_depth": batch.canonical_depth,
            "n_per_depth": batch.n_per_depth,
            "seq_len": batch.seq_len,
            "bracket_types": 1,
        },
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A no-mechanism strawman with the exact `model_fn` signature.

    Returns random causal attention (renormalised in `_normalise_causal`). Used
    by the pipeline smoke test and as an honest 'no circuit' reference: it
    should land near the uniform baseline, far below a real matching head."""
    rng = np.random.default_rng(0)

    def _fn(tokens: np.ndarray) -> np.ndarray:
        length = int(np.asarray(tokens).shape[0])
        return rng.random((length, length))

    return _fn


if __name__ == "__main__":
    payload = evaluate(random_model_fn())
    print("sweep:")
    for r in payload["sweep"]:
        print(r)
