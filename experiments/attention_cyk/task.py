"""Task: can a mechanism place attention on the CYK-correct split point?

CYK (Cocke-Younger-Kasami) parses a string under a CNF grammar by filling a
chart: chart[i][j] = the set of nonterminals deriving seq[i:j]. A cell of
span >= 2 is filled when there is a *split point* k (i < k < j) and a binary
production P -> L R with L in chart[i][k] and R in chart[k][j].

The contract with an attempt is a `model_fn(seq, i, j)` that, for one chart
cell, returns nonnegative scores over candidate split positions. We restrict
to the valid splits i<k<j, normalise to a distribution, and measure how much
probability mass lands on split points that actually fire a production for
that cell. A perfect CYK mechanism puts all mass on correct splits; a uniform
attention puts (#correct / #valid) there.

Pure Python / NumPy. No torch, no I/O, no network.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------
# Grammar  (Chomsky Normal Form)  --  the Dyck-1 (balanced bracket) language.
# nonterminal ids:  S=0  X=1  L=2  R=3
# terminals:        0 = '(' (open)   1 = ')' (close)
#
#   S -> L X | L R | S S        X -> S R        L -> '('     R -> ')'
#
# A span (i,j) is labelled S iff seq[i:j] is a balanced bracket string. The
# correct split points for an S cell are exactly the balance points -- where a
# prefix returns to bracket-depth 0 (an S S split) or the matching of the
# opening bracket (an L X split). This makes the chart SPARSE: most cells are
# empty, and filled cells admit only a few of their candidate splits, so a
# uniform attention scores well below 1.0 and a real mechanism can beat it.
# --------------------------------------------------------------------------
S, X, L, R = 0, 1, 2, 3

# terminal -> nonterminals that derive it (the length-1 spans)
UNARY: dict[int, tuple[int, ...]] = {
    0: (L,),  # '(' -> L
    1: (R,),  # ')' -> R
}

# binary productions as flat (parent, left, right) triples
BINARY_PRODS: tuple[tuple[int, int, int], ...] = (
    (S, L, X),  # S -> '(' (balanced ')')
    (S, L, R),  # S -> '(' ')'
    (S, S, S),  # S -> S S  (concatenation of two balanced strings)
    (X, S, R),  # X -> (balanced) ')'
)

GRAMMAR_ID = "dyck1_v1"
MIN_LEN = 4
MAX_LEN = 9
NUM_STRINGS = 50


# --------------------------------------------------------------------------
# CYK
# --------------------------------------------------------------------------
def _cyk(seq: tuple[int, ...]) -> list[list[set[int]]]:
    """chart[i][j] = set of nonterminals deriving seq[i:j]."""
    n = len(seq)
    chart: list[list[set[int]]] = [
        [set() for _ in range(n + 1)] for _ in range(n + 1)
    ]
    for i in range(n):
        for nt in UNARY.get(seq[i], ()):
            chart[i][i + 1].add(nt)
    for span in range(2, n + 1):
        for i in range(0, n - span + 1):
            j = i + span
            cell = chart[i][j]
            for k in range(i + 1, j):
                left = chart[i][k]
                right = chart[k][j]
                if not left or not right:
                    continue
                for (parent, l, r) in BINARY_PRODS:
                    if l in left and r in right:
                        cell.add(parent)
    return chart


def _correct_splits(chart: list[list[set[int]]], i: int, j: int) -> set[int]:
    """Split points k (i<k<j) that fire at least one production for cell (i,j)."""
    cell = chart[i][j]
    correct: set[int] = set()
    for k in range(i + 1, j):
        left = chart[i][k]
        right = chart[k][j]
        if not left or not right:
            continue
        for (parent, l, r) in BINARY_PRODS:
            if parent in cell and l in left and r in right:
                correct.add(k)
                break
    return correct


def _has_query_cell(chart: list[list[set[int]]], n: int) -> bool:
    # Query cells must have span >= 3 (>= 2 candidate splits) so that the
    # split point is something a mechanism has to *discover*. Span-2 cells have
    # a single forced split: every model_fn scores them 1.0, so they carry no
    # signal and are excluded.
    for i in range(n):
        for j in range(i + 3, n + 1):
            if chart[i][j]:
                return True
    return False


# --------------------------------------------------------------------------
# Batch / generate
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    strings: tuple[tuple[int, ...], ...]
    grammar: str
    max_len: int


def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed: same seed -> same strings."""
    rng = random.Random(seed)
    strings: list[tuple[int, ...]] = []
    attempts = 0
    while len(strings) < NUM_STRINGS and attempts < 5000:
        attempts += 1
        length = rng.randint(MIN_LEN, MAX_LEN)
        seq = tuple(rng.randint(0, 1) for _ in range(length))
        chart = _cyk(seq)
        if _has_query_cell(chart, length):
            strings.append(seq)
    return Batch(strings=tuple(strings), grammar=GRAMMAR_ID, max_len=MAX_LEN)


# --------------------------------------------------------------------------
# model_fn contract
# --------------------------------------------------------------------------
def random_model_fn():
    """A model_fn with the real signature whose body returns random scores.

    Signature:  model_fn(seq: tuple[int,...], i: int, j: int) -> np.ndarray
    The returned array has shape (len(seq)+1,) of nonnegative scores over
    split positions. Used by the smoke test; behaves like an untrained model
    and so scores at roughly the uniform baseline.
    """
    rng = np.random.default_rng(0)

    def fn(seq: tuple[int, ...], i: int, j: int) -> np.ndarray:
        n = len(seq)
        return rng.random(n + 1)

    return fn


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def evaluate(model_fn) -> dict:
    """Run `model_fn` over every query cell, return the benchmark payload."""
    batch = generate()
    per_len: dict[int, dict[str, float]] = defaultdict(
        lambda: {"num_cells": 0.0, "acc_sum": 0.0, "base_sum": 0.0}
    )

    for seq in batch.strings:
        n = len(seq)
        chart = _cyk(seq)
        for i in range(n):
            for j in range(i + 3, n + 1):  # span >= 3: >= 2 candidate splits
                cell = chart[i][j]
                if not cell:
                    continue
                valid_ks = list(range(i + 1, j))
                if not valid_ks:
                    continue
                correct = _correct_splits(chart, i, j)

                raw = np.asarray(model_fn(seq, i, j), dtype=float).ravel()
                if raw.shape[0] < j:
                    padded = np.zeros(j, dtype=float)
                    padded[: raw.shape[0]] = raw
                    raw = padded
                vals = np.clip(raw[i + 1 : j].astype(float), 0.0, None)
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                total = float(vals.sum())
                if total <= 0.0:
                    probs = np.full(len(valid_ks), 1.0 / len(valid_ks))
                else:
                    probs = vals / total

                mass = float(
                    sum(probs[t] for t, k in enumerate(valid_ks) if k in correct)
                )
                base = len(correct) / len(valid_ks)

                rec = per_len[j - i]
                rec["num_cells"] += 1.0
                rec["acc_sum"] += mass
                rec["base_sum"] += base

    sweep = []
    for span_len in sorted(per_len):
        rec = per_len[span_len]
        c = int(rec["num_cells"])
        sweep.append(
            {
                "span_len": int(span_len),
                "num_cells": c,
                "split_accuracy": (rec["acc_sum"] / c) if c else 0.0,
                "uniform_baseline": (rec["base_sum"] / c) if c else 0.0,
            }
        )

    return {
        "version": 1,
        "grammar": batch.grammar,
        "num_strings": len(batch.strings),
        "max_len": batch.max_len,
        "sweep": sweep,
    }
