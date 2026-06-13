"""Synthetic argmax benchmark for attention heads.

Generates controlled key-query similarity tasks where the ground-truth winner
is known. Evaluates a model_fn that implements the attention head.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: fixed query, keys, values, and ground-truth winner index."""
    q: np.ndarray          # (d,)
    K: np.ndarray          # (N, d)
    V: np.ndarray          # (N, d)
    winner_idx: int        # argmax_i (q · k_i)


ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
# model_fn(q: (d,), K: (N, d), V: (N, d)) -> attn_weights: (N,)
# Returns a probability distribution over N positions (non-negative, sums to 1).


# --- Generator --------------------------------------------------------------

# Fixed canonical configuration
_D = 64
_N = 32
_SEPARATIONS = [0.0, 0.5, 1.0, 2.0, 4.0]
_SEEDS_PER_SLICE = 100
_CANONICAL_SEPARATION = 2.0


def _make_batch(seed: int, separation: float) -> Batch:
    """Deterministic batch for a given seed and separation."""
    rng = np.random.default_rng(seed)

    # Query: unit norm
    q = rng.normal(size=_D).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)

    # Keys: all noise, then control the query-direction similarity.
    K = rng.normal(size=(_N, _D)).astype(np.float32)
    winner_idx = int(rng.integers(0, _N))

    # Remove the query-direction component from EVERY key so each distractor
    # has q · k_i = 0 exactly (the "runner-up fixed at 0" condition). Without
    # this, distractors keep ~N(0,1) similarities and the planted winner is
    # frequently NOT the true argmax, which would penalise a correct head.
    proj = K @ q                          # (N,)
    K = K - np.outer(proj, q)             # now q · k_i = 0 for all i

    # Plant the winner at the desired separation: q · k_winner = separation
    # (since ||q|| = 1, the runner-up similarity is 0 and the gap == separation).
    K[winner_idx] = K[winner_idx] + separation * q

    # Values: random, unused by pure attention but part of signature
    V = rng.normal(size=(_N, _D)).astype(np.float32)

    return Batch(q=q, K=K, V=V, winner_idx=winner_idx)


def generate(seed: int = 0) -> Batch:
    """Generate a single batch at the canonical separation.

    For the full sweep, `evaluate` calls `_make_batch` directly with
    `seed = base + slice_idx * 100 + rep`.
    """
    return _make_batch(seed, _CANONICAL_SEPARATION)


# --- Evaluator --------------------------------------------------------------

def _entropy(weights: np.ndarray) -> float:
    """Shannon entropy in nats. weights must be a valid distribution."""
    # Clip for numerical stability
    w = np.clip(weights, 1e-12, 1.0)
    w = w / w.sum()
    return float(-np.sum(w * np.log(w)))


def _rank_of_winner(weights: np.ndarray, winner_idx: int) -> int:
    """1-based rank of winner_idx when weights are sorted descending."""
    # argsort descending: highest weight gets rank 1
    order = np.argsort(weights)[::-1]
    return int(np.where(order == winner_idx)[0][0] + 1)


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over the full separation sweep, return payload for benchmark.score."""
    sweep_records = []

    for slice_idx, sep in enumerate(_SEPARATIONS):
        winner_mass_vals = []
        winner_rank_vals = []
        entropy_vals = []

        for rep in range(_SEEDS_PER_SLICE):
            # Deterministic seed per (slice, rep)
            seed = slice_idx * _SEEDS_PER_SLICE + rep
            batch = _make_batch(seed, sep)

            # Call the model
            attn = model_fn(batch.q, batch.K, batch.V)

            # Validate output shape and normalization
            if attn.shape != (_N,):
                raise ValueError(f"model_fn returned shape {attn.shape}, expected ({_N},)")
            if not np.all(attn >= -1e-6):
                raise ValueError("model_fn returned negative attention weights")
            if abs(attn.sum() - 1.0) > 1e-4:
                raise ValueError(f"model_fn weights sum to {attn.sum():.6f}, expected 1.0")

            # Record measurements
            winner_mass_vals.append(float(attn[batch.winner_idx]))
            winner_rank_vals.append(_rank_of_winner(attn, batch.winner_idx))
            entropy_vals.append(_entropy(attn))

        sweep_records.append({
            "separation": sep,
            "winner_mass_mean": float(np.mean(winner_mass_vals)),
            "winner_mass_std": float(np.std(winner_mass_vals, ddof=1)),
            "winner_rank_mean": float(np.mean(winner_rank_vals)),
            "winner_rank_std": float(np.std(winner_rank_vals, ddof=1)),
            "entropy_mean": float(np.mean(entropy_vals)),
            "entropy_std": float(np.std(entropy_vals, ddof=1)),
        })

    payload = {
        "version": 1,
        "config": {
            "d": _D,
            "N": _N,
            "separations": _SEPARATIONS,
            "seeds_per_slice": _SEEDS_PER_SLICE,
            "canonical_separation": _CANONICAL_SEPARATION,
        },
        "sweep": sweep_records,
        "baselines": {
            "uniform_winner_mass": 1.0 / _N,
            "uniform_entropy": math.log(_N),
        },
    }
    return payload


# --- Random model_fn for smoke test ----------------------------------------

def random_model_fn() -> ModelFn:
    """Returns a model_fn that outputs a uniform distribution (valid but uninformative)."""
    def _uniform(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
        N = K.shape[0]
        return np.ones(N, dtype=np.float32) / N
    return _uniform