"""Data generation and evaluation for attention_prefix_sum."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

# ModelFn signature: input_ids [B, L] -> logits [B, L, V]
ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray          # [B, L], int32, values in 0..vocab_size-1
    target_ids: np.ndarray         # [B, L], int32, prefix sums mod vocab_size
    seq_len: int
    vocab_size: int


def _prefix_sum_mod(arr: np.ndarray, mod: int) -> np.ndarray:
    """Cumulative sum mod `mod` along last axis."""
    return np.cumsum(arr, axis=-1) % mod


def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch for the canonical condition.
    Returns a single batch with seq_len=16, vocab_size=10, 512 sequences.
    """
    rng = np.random.default_rng(seed)
    vocab_size = 10
    seq_len = 16
    num_sequences = 512

    # Random input tokens
    input_ids = rng.integers(0, vocab_size, size=(num_sequences, seq_len), dtype=np.int32)
    # Target is prefix sum mod vocab_size at each position
    target_ids = _prefix_sum_mod(input_ids, vocab_size)

    return Batch(
        input_ids=input_ids,
        target_ids=target_ids,
        seq_len=seq_len,
        vocab_size=vocab_size,
    )


def _evaluate_on_batch(model_fn: ModelFn, batch: Batch) -> tuple[int, int]:
    """Run model_fn on a batch, return (correct, total) token predictions."""
    logits = model_fn(batch.input_ids)                     # [B, L, V]
    if logits.shape != (batch.input_ids.shape[0], batch.seq_len, batch.vocab_size):
        raise ValueError(
            f"model_fn returned logits shape {logits.shape}, "
            f"expected {(batch.input_ids.shape[0], batch.seq_len, batch.vocab_size)}"
        )
    pred_ids = logits.argmax(axis=-1).astype(np.int32)     # [B, L]
    correct = int((pred_ids == batch.target_ids).sum())
    total = int(batch.target_ids.size)
    return correct, total


def evaluate(model_fn: ModelFn) -> dict:
    """
    Evaluate model_fn across the length sweep [4, 8, 16, 32, 64].
    Returns payload dict matching benchmark.score contract.
    """
    vocab_size = 10
    canonical_seq_len = 16
    num_sequences = 512
    sweep_lengths = [4, 8, 16, 32, 64]
    seed = 0

    rng = np.random.default_rng(seed)
    sweep = []

    for seq_len in sweep_lengths:
        # Generate fresh batch for this length (deterministic per length)
        # Use a derived seed so each length gets different but reproducible data
        length_seed = seed * 1000 + seq_len
        length_rng = np.random.default_rng(length_seed)

        input_ids = length_rng.integers(0, vocab_size, size=(num_sequences, seq_len), dtype=np.int32)
        target_ids = _prefix_sum_mod(input_ids, vocab_size)

        batch = Batch(
            input_ids=input_ids,
            target_ids=target_ids,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )

        correct, total = _evaluate_on_batch(model_fn, batch)
        sweep.append({"seq_len": seq_len, "correct": correct, "total": total})

    # Canonical condition metrics (seq_len=16)
    canonical_record = next(r for r in sweep if r["seq_len"] == canonical_seq_len)
    random_baseline_total = canonical_record["total"]
    random_baseline_correct = random_baseline_total // vocab_size  # analytic expectation

    return {
        "version": 1,
        "config": {
            "seq_len": canonical_seq_len,
            "vocab_size": vocab_size,
            "num_sequences": num_sequences,
            "seed": seed,
        },
        "sweep": sweep,
        "random_baseline_correct": random_baseline_correct,
        "random_baseline_total": random_baseline_total,
    }


def random_model_fn() -> ModelFn:
    """
    Factory returning a model_fn for smoke testing.

    The returned callable has exactly the same signature as a real model_fn:
        model_fn(input_ids: np.ndarray[B, L]) -> logits: np.ndarray[B, L, V]
    Its body returns zero logits (uniform distribution -> random guessing),
    matching the contract shape without any torch/GPU dependency.
    """
    vocab_size = 10

    def _random_fn(input_ids: np.ndarray) -> np.ndarray:
        batch, seq_len = input_ids.shape
        # Zeros -> uniform distribution -> random guessing
        return np.zeros((batch, seq_len, vocab_size), dtype=np.float32)

    return _random_fn