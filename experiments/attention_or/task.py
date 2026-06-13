"""Task definition for the `attention_or` goal.

The goal owns this file just like it owns `benchmark.py`. Every attempt
imports it so the cosine-similarity sweep, the canonical ``d`` / canonical
``rho``, the query/key/value geometry, and the Boolean truth table are
byte-identical across attempts — no drift in what the question actually is.

An attempt only contributes a **model function**:

    model_fn(batch: Batch) -> np.ndarray   # shape (4, d)

returning the attention-output vector for each of the four Boolean input
pairs, in the exact order of ``batch.inputs`` — i.e.
``(0,0), (0,1), (1,0), (1,1)``. The attempt implements the forward pass of
the 1-head block; the framework only supplies the geometry (``q_A``, ``q_B``,
``k_A``, ``k_B``, ``v_A``, ``v_B``) and the input pairs.

`evaluate` builds one batch per ``rho`` in the canonical sweep, runs the
model, computes per-slice sharpness, and assembles the payload that
`benchmark.score()` consumes. Hand-built attempts can call `evaluate`
directly with no training; trained attempts call it after training. Both
paths produce the same payload shape.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Keep in sync with `benchmark.VERSION`. The benchmark validates this on the
# payload and raises if they drift, so a missed bump fails loudly rather than
# silently producing wrong numbers.
VERSION: int = 1

D: int = 32
CANONICAL_RHO: float = 0.7

# Cosine similarity sweep between the two feature query vectors. Exact
# two-decimal values so the sweep keys never carry float dust. Must match the
# README "Payload contract" and the per-slice metric tags in `benchmark.py`.
RHO_SWEEP: list[float] = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95]


@dataclass(frozen=True)
class Batch:
    """One synthetic 1-head attention configuration at a single ``rho``.

    Vectors are plain ``np.ndarray`` of shape ``(d,)``. ``inputs`` is the
    fixed truth table in canonical order; ``model_fn`` must return outputs in
    the same order.
    """

    rho: float
    d: int
    q_A: np.ndarray
    q_B: np.ndarray
    k_A: np.ndarray
    k_B: np.ndarray
    v_A: np.ndarray
    v_B: np.ndarray
    inputs: list[tuple[int, int]]


# model_fn(batch) -> np.ndarray of shape (4, d), rows aligned to batch.inputs.
ModelFn = Callable[[Batch], np.ndarray]

# Canonical truth table order: (0,0), (0,1), (1,0), (1,1).
_INPUTS: list[tuple[int, int]] = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_batch(rho: float, d: int, seed: int) -> Batch:
    """Build a deterministic batch whose two query vectors have ``cos = rho``.

    Construction (matches README "Setup"):
      - ``q_A`` is a fixed unit vector.
      - ``q_B = rho * q_A + sqrt(1 - rho^2) * u`` with ``u`` a unit vector
        orthogonal to ``q_A``, so ``cos(q_A, q_B) == rho`` exactly.
      - ``k_A = q_A``, ``k_B = q_B`` (matched queries/keys).
      - ``v_A = v_B = [1, 0, ..., 0]`` (scalar 1 in the first component).
    """
    rho = float(np.clip(rho, -1.0, 1.0))
    rng = np.random.default_rng(seed)

    q_A = _unit(rng.standard_normal(d))
    u = rng.standard_normal(d)
    u = u - (u @ q_A) * q_A  # orthogonalise against q_A
    u = _unit(u)
    q_B = _unit(rho * q_A + math.sqrt(max(0.0, 1.0 - rho * rho)) * u)

    e0 = np.zeros(d, dtype=float)
    e0[0] = 1.0

    return Batch(
        rho=rho,
        d=d,
        q_A=q_A,
        q_B=q_B,
        k_A=q_A.copy(),
        k_B=q_B.copy(),
        v_A=e0.copy(),
        v_B=e0.copy(),
        inputs=list(_INPUTS),
    )


def generate(seed: int = 0) -> Batch:
    """Return the canonical batch (at ``rho = CANONICAL_RHO``).

    Deterministic for a given seed: same seed → byte-identical batch. The
    geometry is the single source of truth for the question; attempts may not
    change it — they only provide ``model_fn``. ``seed`` controls the fixed
    random query directions; the canonical condition uses ``seed = 0``.
    """
    return _make_batch(CANONICAL_RHO, D, seed)


def evaluate(model_fn: ModelFn, *, seed: int = 0) -> dict[str, Any]:
    """Run `model_fn` across the canonical cosine sweep, return a benchmark payload.

    Builds one batch per ``rho`` in ``RHO_SWEEP`` (same ``seed`` so the query
    directions are shared and only their relative cosine changes), runs the
    model, and assembles the dict that `benchmark.score` consumes — pass it
    straight to `record_benchmark`.

    Raises:
        ValueError: if `model_fn` returns the wrong shape or a non-finite value.
    """
    sweep: list[dict[str, Any]] = []
    for rho in RHO_SWEEP:
        batch = _make_batch(rho, D, seed)
        out = np.asarray(model_fn(batch), dtype=float)
        if out.shape != (4, D):
            raise ValueError(
                f"model_fn(rho={rho}) returned shape {out.shape}, expected (4, {D})"
            )
        if not np.all(np.isfinite(out)):
            raise ValueError(f"model_fn(rho={rho}) returned non-finite values")

        out_00, out_01, out_10, out_11 = (out[0], out[1], out[2], out[3])

        # Sharpness uses the first component only (the value basis). Identical
        # formula to benchmark.score so the precomputed value matches its
        # recomputation exactly.
        or1 = [float(out_01[0]), float(out_10[0]), float(out_11[0])]
        mean_or1 = sum(or1) / 3.0
        gap = mean_or1 - float(out_00[0])
        denom = (max(or1) - min(or1)) + 1e-8
        sharpness = gap / denom

        sweep.append(
            {
                "rho": rho,
                "out_00": out_00.tolist(),
                "out_01": out_01.tolist(),
                "out_10": out_10.tolist(),
                "out_11": out_11.tolist(),
                "sharpness": sharpness,
            }
        )

    return {
        "version": VERSION,
        "config": {
            "d": D,
            "canonical_rho": CANONICAL_RHO,
            "rho_sweep": list(RHO_SWEEP),
        },
        "sweep": sweep,
    }
