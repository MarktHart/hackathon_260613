"""
Synthetic k-th position selection task.

generate(seed) -> Batch
evaluate(model_fn) -> payload dict
random_model_fn() -> model_fn returning uniform attention
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray      # (B, L), int32
    positions: np.ndarray      # (L,), int32
    target_k: int              # the k this batch tests


# Fixed canonical sweep — do not change without bumping VERSION
SWEEP_K = [0, 4, 8, 12, 16, 20, 24, 28]
CANONICAL_K = 8
L = 32
V = 100
MARKER = 99
BATCH_SIZE = 128


def generate(seed: int = 0) -> list[Batch]:
    """
    Deterministic batch generator for the full sweep.
    Returns a list of 8 Batches, one per k in SWEEP_K.
    """
    rng = np.random.default_rng(seed)
    batches = []

    for k in SWEEP_K:
        # Random noise tokens everywhere, drawn from the FULL vocab 0..V-1
        # (including the marker value). This is deliberate: position k must not
        # be uniquely identifiable by content, so a content-matching head cannot
        # shortcut the positional task. See README "Setup".
        input_ids = rng.integers(0, V, size=(BATCH_SIZE, L), dtype=np.int32)
        # Place marker at position k
        input_ids[:, k] = MARKER
        positions = np.arange(L, dtype=np.int32)
        batches.append(Batch(input_ids=input_ids, positions=positions, target_k=k))

    return batches


def evaluate(model_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> dict:
    """
    Run model_fn on each batch in the sweep, collect attention statistics.
    model_fn(input_ids, positions) -> attn_weights (B, L), non-negative, sums to 1 over L.
    """
    batches = generate(seed=0)  # canonical seed fixed
    sweep_records = []

    for batch in batches:
        attn = model_fn(batch.input_ids, batch.positions)  # (B, L)
        # Validate shape and normalization
        assert attn.shape == (BATCH_SIZE, L), f"attn shape {attn.shape} != ({BATCH_SIZE}, {L})"
        # Allow small numerical drift
        sums = attn.sum(axis=1)
        assert np.allclose(sums, 1.0, atol=1e-4), f"attention doesn't sum to 1: {sums[:5]}"

        k = batch.target_k
        attn_at_k = float(attn[:, k].mean())
        # Entropy in nats
        eps = 1e-12
        entropy = float(-np.mean(np.sum(attn * np.log(attn + eps), axis=1)))
        # Position of max attention per sequence, then mean
        max_pos = float(np.mean(np.argmax(attn, axis=1).astype(float)))

        sweep_records.append({
            "k": int(k),
            "attn_at_k": attn_at_k,
            "attn_entropy": entropy,
            "attn_max_pos": max_pos,
            "batch_size": BATCH_SIZE,
        })

    return {
        "version": 1,
        "canonical_k": CANONICAL_K,
        "sweep": sweep_records,
        "model_name": "unknown",  # attempt overwrites this
        "dataset": "synthetic_kth_select",
    }


def random_model_fn() -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """
    Returns a model_fn that outputs uniform attention (1/L) for every position.
    Pure NumPy, no torch, no GPU.
    """
    def _uniform(input_ids: np.ndarray, positions: np.ndarray) -> np.ndarray:
        B = input_ids.shape[0]
        L = input_ids.shape[1]
        return np.full((B, L), 1.0 / L, dtype=np.float32)

    return _uniform