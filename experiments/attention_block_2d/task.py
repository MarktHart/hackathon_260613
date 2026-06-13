"""Synthetic 2D attention pattern generation and evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Batch:
    """Container for a fixed set of attention matrices and their ground truth."""
    matrices: np.ndarray          # shape (n_examples, N, N)
    pattern_ids: list[str]        # length n_examples
    params: list[dict]            # length n_examples


# -----------------------------------------------------------------------------
# Pattern generators
# -----------------------------------------------------------------------------

def _make_local(N: int, H: int, W: int, window_size: int) -> np.ndarray:
    """Local square window attention (radius = window_size)."""
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        r_q, c_q = divmod(i, W)
        allowed = []
        for dr in range(-window_size, window_size + 1):
            for dc in range(-window_size, window_size + 1):
                r_k, c_k = r_q + dr, c_q + dc
                if 0 <= r_k < H and 0 <= c_k < W:
                    allowed.append(r_k * W + c_k)
        A[i, allowed] = 1.0
    return A


def _make_dilated(N: int, H: int, W: int, window_size: int, dilation: int) -> np.ndarray:
    """Dilated window attention."""
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        r_q, c_q = divmod(i, W)
        allowed = []
        for dr in range(-window_size, window_size + 1):
            for dc in range(-window_size, window_size + 1):
                r_k, c_k = r_q + dr * dilation, c_q + dc * dilation
                if 0 <= r_k < H and 0 <= c_k < W:
                    allowed.append(r_k * W + c_k)
        A[i, allowed] = 1.0
    return A


def _make_global(N: int, global_pos: int) -> np.ndarray:
    """One global token attends to all; all attend to global."""
    A = np.zeros((N, N), dtype=np.float32)
    # global token attends to everyone uniformly
    A[global_pos, :] = 1.0
    # everyone attends to global token
    A[:, global_pos] = 1.0
    # other rows: only global token (already set by column)
    return A


def _make_causal_2d(N: int) -> np.ndarray:
    """Raster-order causal: each position attends to itself and all earlier positions."""
    A = np.tril(np.ones((N, N), dtype=np.float32))
    return A


def _normalise_rows(A: np.ndarray, rng: np.random.Generator, eps: float = 1e-3) -> np.ndarray:
    """Add i.i.d. uniform noise in [0, eps] and renormalise rows to sum to 1.

    Drawing the noise from `rng` makes each realisation distinct and
    seed-dependent while keeping the underlying structure recoverable
    (allowed keys carry mass ~1, disallowed keys carry mass <= eps).
    """
    N = A.shape[0]
    A = A + eps * rng.random((N, N), dtype=np.float32)
    row_sums = A.sum(axis=1, keepdims=True)
    return A / row_sums


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch of 16 attention matrices (4 patterns × 4 noise
    realisations each, all drawn from the single `seed`-seeded RNG stream).
    Canonical grid: 8×8 (N=64). Canonical parameters per pattern family.
    """
    rng = np.random.default_rng(seed)
    H, W = 8, 8
    N = H * W

    # Canonical parameter for each pattern family
    canonical = {
        "local": {"window_size": 1},
        "dilated": {"window_size": 1, "dilation": 2},
        "global": {"global_pos": 0},
        "causal_2d": {},
    }

    matrices = []
    pattern_ids = []
    params_list = []

    for pattern_id, base_params in canonical.items():
        for _ in range(4):  # 4 examples per pattern
            if pattern_id == "local":
                A = _make_local(N, H, W, base_params["window_size"])
            elif pattern_id == "dilated":
                A = _make_dilated(N, H, W, base_params["window_size"], base_params["dilation"])
            elif pattern_id == "global":
                A = _make_global(N, base_params["global_pos"])
            elif pattern_id == "causal_2d":
                A = _make_causal_2d(N)
            else:
                raise ValueError(f"Unknown pattern_id: {pattern_id}")

            A = _normalise_rows(A, rng, eps=1e-3)
            matrices.append(A)
            pattern_ids.append(pattern_id)
            params_list.append(dict(base_params))  # copy

    return Batch(
        matrices=np.stack(matrices, axis=0),  # (16, N, N)
        pattern_ids=pattern_ids,
        params=params_list,
    )


def evaluate(model_fn: Callable[[np.ndarray], dict]) -> dict:
    """
    Run `model_fn` on every matrix in the canonical batch (seed=0).
    Returns payload dict matching benchmark.score's expected shape.
    """
    batch = generate(seed=0)
    n = batch.matrices.shape[0]
    sweep = []

    for i in range(n):
        attn = batch.matrices[i]  # (N, N)
        pred = model_fn(attn)

        # Validate prediction structure
        pred_pattern = pred.get("pattern_id", "")
        pred_params = pred.get("params", {})
        confidence = float(pred.get("confidence", 0.0))
        confidence = max(0.0, min(1.0, confidence))  # clamp

        # Correct iff pattern matches AND every ground-truth parameter value
        # is matched. Extra keys in pred_params are ignored (see README).
        gt_params = batch.params[i]
        params_match = all(pred_params.get(k) == v for k, v in gt_params.items())
        correct = (pred_pattern == batch.pattern_ids[i]) and params_match

        sweep.append({
            "pattern_id": batch.pattern_ids[i],
            "params": batch.params[i],
            "pred_pattern_id": pred_pattern,
            "pred_params": pred_params,
            "confidence": confidence,
            "correct": correct,
        })

    return {
        "version": 1,
        "grid_size": (8, 8),
        "sweep": sweep,
    }


def random_model_fn() -> Callable[[np.ndarray], dict]:
    """
    Returns a callable with the exact ModelFn signature that emits
    random-but-valid predictions. Pure NumPy, no torch, no GPU.
    """
    pattern_ids = ["local", "dilated", "global", "causal_2d"]
    param_options = {
        "local": [{"window_size": 1}, {"window_size": 2}, {"window_size": 3}],
        "dilated": [
            {"window_size": 1, "dilation": 1},
            {"window_size": 1, "dilation": 2},
            {"window_size": 2, "dilation": 1},
        ],
        "global": [{"global_pos": 0}, {"global_pos": 63}],
        "causal_2d": [{}],
    }

    rng = np.random.default_rng(12345)  # fixed seed for determinism

    def _fn(attn: np.ndarray) -> dict:
        # Randomly pick a pattern and its params
        pid = rng.choice(pattern_ids)
        params = rng.choice(param_options[pid])
        confidence = float(rng.uniform(0.1, 0.9))
        return {
            "pattern_id": pid,
            "params": dict(params),
            "confidence": confidence,
        }

    return _fn