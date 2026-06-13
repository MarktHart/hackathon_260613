"""Task for the attention_range_sum goal.

Synthetic, deterministic. Pure Python / NumPy. No I/O, no network.

A range-sum query asks: given a fixed token sequence and a contiguous window
[start, end), what is the sum of the token values inside that window? The data
is grouped into slices by the *range length* k = end - start so the benchmark
can chart how a model degrades as the window grows.

Exports
-------
generate(seed) -> Batch        deterministic data
evaluate(model_fn) -> payload  runs model_fn, returns benchmark-ready payload
random_model_fn() -> ModelFn   a correctly-shaped dummy model (smoke test)

The model_fn contract (the goal's interface with attempts):

    model_fn(input_ids: np.ndarray, start: int, end: int) -> float

    `input_ids` is the full sequence (shape (L,), int values in [0, V)).
    `start`, `end` define the half-open window [start, end). The function
    returns a single scalar: the predicted sum of input_ids[start:end].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Slices of the sweep — must match benchmark.RANGE_LENS exactly.
RANGE_LENS: list[int] = [2, 4, 8, 16, 32]
QUERIES_PER_LEN: int = 200
SEQ_LEN: int = 64
VOCAB_SIZE: int = 10
CANONICAL_SEED: int = 42

ModelFn = Callable[[np.ndarray, int, int], float]


@dataclass(frozen=True)
class Batch:
    """A deterministic evaluation batch.

    queries / targets are grouped by range length: queries[k] is an array of
    shape (QUERIES_PER_LEN, 2) of (start, end) rows, and targets[k] is the
    matching (QUERIES_PER_LEN,) array of true sums.
    """

    input_ids: np.ndarray                 # shape (L,), values in [0, V)
    queries: dict                         # k -> (QUERIES_PER_LEN, 2) int array
    targets: dict                         # k -> (QUERIES_PER_LEN,) int array
    config: dict                          # generation config for reproducibility


def generate(seed: int = CANONICAL_SEED) -> Batch:
    """Deterministic batch of range-sum queries grouped by range length.

    Same seed -> same batch. The canonical condition uses seed=42, SEQ_LEN=64,
    VOCAB_SIZE=10, with QUERIES_PER_LEN windows sampled for each k in
    RANGE_LENS.
    """
    rng = np.random.default_rng(seed)

    input_ids = rng.integers(0, VOCAB_SIZE, size=SEQ_LEN, dtype=np.int64)

    queries: dict = {}
    targets: dict = {}
    for k in RANGE_LENS:
        if k > SEQ_LEN:
            raise ValueError(f"range_len {k} exceeds seq_len {SEQ_LEN}")
        starts = rng.integers(0, SEQ_LEN - k + 1, size=QUERIES_PER_LEN)
        ends = starts + k
        q = np.stack([starts, ends], axis=1).astype(np.int64)
        t = np.array(
            [int(input_ids[s:e].sum()) for s, e in q], dtype=np.int64
        )
        queries[k] = q
        targets[k] = t

    config = {
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "range_lens": list(RANGE_LENS),
        "queries_per_len": QUERIES_PER_LEN,
        "seed": int(seed),
    }

    return Batch(
        input_ids=input_ids,
        queries=queries,
        targets=targets,
        config=config,
    )


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over the canonical batch; return a benchmark-ready payload.

    The payload is a sweep with one record per range length, each carrying the
    flat list of the model's predictions and the matching ground-truth sums.
    """
    batch = generate(seed=CANONICAL_SEED)

    sweep = []
    for k in RANGE_LENS:
        q = batch.queries[k]
        t = batch.targets[k]
        preds: list[float] = []
        for (start, end) in q:
            preds.append(float(model_fn(batch.input_ids, int(start), int(end))))
        sweep.append(
            {
                "range_len": int(k),
                "predictions": preds,
                "targets": [int(v) for v in t],
            }
        )

    return {
        "version": 1,
        "config": batch.config,
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A correctly-shaped dummy model: random predictions, no torch, no GPU.

    Signature matches a real model_fn exactly:
        (input_ids, start, end) -> float
    Used by the pipeline smoke test to validate the payload contract.
    """
    rng = np.random.default_rng(12345)

    def _random_predict(input_ids: np.ndarray, start: int, end: int) -> float:
        range_len = int(end) - int(start)
        max_possible = range_len * (VOCAB_SIZE - 1)
        return float(rng.uniform(0.0, max(1.0, max_possible)))

    return _random_predict
