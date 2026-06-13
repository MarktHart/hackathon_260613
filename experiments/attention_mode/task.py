"""
Synthetic *attention mode* classification task.

A "mode" is a canonical, human-nameable attention pattern (positional,
uniform, diagonal, induction, previous-token). The goal asks: given a head's
raw attention matrix, can an attempt's mechanism name the mode it implements,
and how gracefully does that judgement degrade as the pattern is corrupted by
noise?

This module owns the data and the evaluation. Attempts never construct the
payload themselves: they hand `evaluate` a `model_fn` and receive a
ready-to-record payload that `benchmark.score` consumes verbatim.

Pure NumPy. No torch, no GPU, no I/O.

`model_fn` contract
-------------------
    model_fn(attention_matrices: np.ndarray) -> np.ndarray

    input : (n_heads, L, L) float32, each row of each (L, L) matrix is a
            probability distribution over keys (sums to 1).
    output: (n_heads, N_MODES) float32, each row a probability distribution
            over MODES *in MODES order*, summing to 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
class ModelFn(Protocol):
    def __call__(self, attention_matrices: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Batch:
    noise: float                      # corruption level applied to this batch
    attention_matrices: np.ndarray    # (n_heads, L, L) float32, rows sum to 1
    true_modes: list[str]             # length n_heads, each in MODES


# ----------------------------------------------------------------------
# Constants — the canonical measurement condition
# ----------------------------------------------------------------------
MODES = ("positional", "uniform", "diagonal", "induction", "previous_token")
MODE_TO_IDX = {m: i for i, m in enumerate(MODES)}
N_MODES = len(MODES)

CANONICAL_L = 16          # sequence length
N_PER_MODE = 10           # heads generated per mode at each noise level
CANONICAL_SEED = 0

# Sweep axis: how much uniform noise is mixed into the clean pattern.
# noise=0.0 is the canonical (clean) condition.
NOISE_LEVELS = (0.0, 0.1, 0.2, 0.3, 0.5)
CANONICAL_NOISE = 0.0


# ----------------------------------------------------------------------
# Clean pattern generators (deterministic, rows sum to 1)
# ----------------------------------------------------------------------
def _positional_pattern(L: int, anchor: int = 0) -> np.ndarray:
    """All queries attend to a single fixed key position."""
    pat = np.zeros((L, L), dtype=np.float32)
    pat[:, anchor % L] = 1.0
    return pat


def _uniform_pattern(L: int) -> np.ndarray:
    """Uniform attention over all keys."""
    return np.full((L, L), 1.0 / L, dtype=np.float32)


def _diagonal_pattern(L: int) -> np.ndarray:
    """Each query attends to its own index (i -> i)."""
    return np.eye(L, dtype=np.float32)


def _induction_pattern(L: int) -> np.ndarray:
    """Each query attends to the next position (i -> i+1, fixed offset)."""
    pat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        pat[i, (i + 1) % L] = 1.0
    return pat


def _previous_token_pattern(L: int) -> np.ndarray:
    """Each query attends to the previous position (i -> i-1)."""
    pat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        pat[i, (i - 1) % L] = 1.0
    return pat


_PATTERN_FNS = {
    "positional": _positional_pattern,
    "uniform": _uniform_pattern,
    "diagonal": _diagonal_pattern,
    "induction": _induction_pattern,
    "previous_token": _previous_token_pattern,
}


def _apply_noise(pat: np.ndarray, noise: float, rng: np.random.Generator) -> np.ndarray:
    """
    Convex-combine the clean pattern with a random row-stochastic matrix:
        out = (1 - noise) * pat + noise * random_simplex
    Output rows still sum to 1.
    """
    if noise <= 0.0:
        return pat.astype(np.float32)
    L = pat.shape[0]
    rand = rng.exponential(scale=1.0, size=(L, L)).astype(np.float32)
    rand /= rand.sum(axis=1, keepdims=True)
    out = (1.0 - noise) * pat + noise * rand
    # Renormalise defensively against float drift.
    out /= out.sum(axis=1, keepdims=True)
    return out.astype(np.float32)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------
def generate(seed: int = CANONICAL_SEED, noise: float = CANONICAL_NOISE) -> Batch:
    """
    Deterministic for a given (seed, noise): same arguments -> same Batch.

    Produces N_PER_MODE heads for each mode in MODES (so N_PER_MODE * N_MODES
    heads total), at the requested noise level.
    """
    # Derive a stable per-(seed, noise) stream without relying on global state.
    noise_key = int(round(noise * 1000))
    rng = np.random.default_rng((seed + 1) * 1_000_003 + noise_key)

    mats: list[np.ndarray] = []
    true_modes: list[str] = []
    for mode in MODES:
        clean = _PATTERN_FNS[mode](CANONICAL_L)
        for _ in range(N_PER_MODE):
            mats.append(_apply_noise(clean, noise, rng))
            true_modes.append(mode)

    attention_matrices = np.stack(mats, axis=0).astype(np.float32)
    return Batch(noise=float(noise), attention_matrices=attention_matrices, true_modes=true_modes)


def _coerce_probs(pred: np.ndarray, n_heads: int) -> np.ndarray:
    """Validate and normalise a model_fn output into (n_heads, N_MODES)."""
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != (n_heads, N_MODES):
        raise ValueError(
            f"model_fn returned shape {pred.shape}, expected ({n_heads}, {N_MODES})"
        )
    if not np.all(np.isfinite(pred)):
        raise ValueError("model_fn returned non-finite probabilities")
    if np.any(pred < -1e-6):
        raise ValueError("model_fn returned negative probabilities")
    pred = np.clip(pred, 0.0, None)
    sums = pred.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("model_fn returned an all-zero probability row")
    return pred / sums


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run `model_fn` over every noise level in NOISE_LEVELS and return the
    payload that `benchmark.score` consumes.
    """
    sweep: list[dict] = []
    for noise in NOISE_LEVELS:
        batch = generate(CANONICAL_SEED, noise)
        n_heads = len(batch.true_modes)
        probs = _coerce_probs(model_fn(batch.attention_matrices), n_heads)
        for i in range(n_heads):
            sweep.append({
                "noise": float(noise),
                "true_mode": batch.true_modes[i],
                "pred_probs": {m: float(probs[i, j]) for j, m in enumerate(MODES)},
            })

    payload = {
        "version": 1,
        "L": CANONICAL_L,
        "seed": CANONICAL_SEED,
        "modes": list(MODES),
        "noise_levels": list(NOISE_LEVELS),
        "canonical_noise": CANONICAL_NOISE,
        "n_per_mode": N_PER_MODE,
        "sweep": sweep,
    }
    return payload


def random_model_fn() -> ModelFn:
    """
    A baseline `model_fn`: returns uniform-random probability rows
    (Dirichlet(1, ..., 1)) of the correct shape. Pure NumPy, no torch.
    Used by the pipeline smoke test.
    """
    def _fn(attention_matrices: np.ndarray) -> np.ndarray:
        n_heads = int(np.asarray(attention_matrices).shape[0])
        rng = np.random.default_rng()
        samples = rng.exponential(scale=1.0, size=(n_heads, N_MODES))
        probs = samples / samples.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    return _fn
