"""Task: can a model compute Conway's Game of Life next-state from a board?

This is the shared, model-agnostic data + evaluation harness for the
`attention_game_of_life` goal. Every attempt imports `evaluate` and hands it a
`model_fn`; it never constructs the payload itself.

model_fn contract
-----------------
    model_fn(grids: np.ndarray) -> np.ndarray

    * input  `grids`  : float32 array of shape (B, H, W), values in {0.0, 1.0}
                        (current board states).
    * output `logits` : float array of shape (B, H, W). Interpreted as a
                        per-cell LOGIT for "cell is alive in the next state".
                        A cell is predicted alive iff its logit is > 0.

The mapping from input board to the true next board is fixed Conway's Game of
Life with toroidal (wrap-around) boundaries — see `_next_state`.
"""

from dataclasses import dataclass

import numpy as np

# --- Canonical measurement condition -------------------------------------
B, H, W = 32, 16, 16
DENSITIES = (0.1, 0.2, 0.3, 0.4, 0.5)   # initial live-cell fraction sweep
CANONICAL_DENSITY = 0.3


@dataclass(frozen=True)
class Batch:
    """Deterministic boards + ground-truth next states, one entry per density."""
    densities: tuple
    canonical_density: float
    grids: dict          # density -> (B, H, W) float32 current states {0,1}
    labels: dict         # density -> (B, H, W) int32   next states {0,1}
    height: int
    width: int
    batch_size: int
    seed: int


def _next_state(grids: np.ndarray) -> np.ndarray:
    """Conway's Game of Life next state for a batch of grids.

    Toroidal (periodic) boundary conditions.
    """
    _, h, w = grids.shape
    padded = np.pad(grids, ((0, 0), (1, 1), (1, 1)), mode="wrap")
    neighbor_counts = np.zeros_like(grids, dtype=np.int32)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue
            neighbor_counts += padded[:, di:di + h, dj:dj + w].astype(np.int32)
    alive = grids.astype(np.int32)
    survive = (alive == 1) & ((neighbor_counts == 2) | (neighbor_counts == 3))
    birth = (alive == 0) & (neighbor_counts == 3)
    return (survive | birth).astype(np.int32)


def generate(seed: int = 0) -> Batch:
    """Deterministic batch of random boards and their true GoL next states.

    Same seed -> same boards. A distinct sub-stream is drawn per density so
    adding/removing a density does not perturb the others.
    """
    grids: dict = {}
    labels: dict = {}
    for i, density in enumerate(DENSITIES):
        rng = np.random.default_rng([seed, i])
        g = (rng.random((B, H, W)) < density).astype(np.float32)
        grids[density] = g
        labels[density] = _next_state(g)
    return Batch(
        densities=DENSITIES,
        canonical_density=CANONICAL_DENSITY,
        grids=grids,
        labels=labels,
        height=H,
        width=W,
        batch_size=B,
        seed=seed,
    )


def _counts(pred_live: np.ndarray, true_live: np.ndarray) -> dict:
    """Confusion counts for the 'alive in next state' positive class."""
    pred_live = pred_live.astype(bool)
    true_live = true_live.astype(bool)
    tp = int(np.sum(pred_live & true_live))
    fp = int(np.sum(pred_live & ~true_live))
    fn = int(np.sum(~pred_live & true_live))
    tn = int(np.sum(~pred_live & ~true_live))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def evaluate(model_fn) -> dict:
    """Run `model_fn` over every density slice and return the scored payload.

    The payload is pre-aggregated to integer confusion counts per slice; the
    benchmark derives accuracy / F1 / baselines and handles zero denominators.
    """
    batch = generate(seed=0)

    sweep = []
    for density in batch.densities:
        grids = batch.grids[density]
        truth = batch.labels[density]

        logits = model_fn(grids)
        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != grids.shape:
            raise ValueError(
                f"model_fn output shape {logits.shape} != grid shape {grids.shape} "
                f"(density={density})"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError(f"model_fn produced non-finite logits (density={density})")

        pred_live = logits > 0.0

        # Static baseline: predict next == current board.
        static_pred = grids > 0.5

        model_c = _counts(pred_live, truth)
        static_c = _counts(static_pred, truth)

        n_cells = int(grids.size)
        n_correct = int(np.sum(pred_live == (truth > 0)))
        static_correct = int(np.sum(static_pred == (truth > 0)))

        sweep.append({
            "density": float(density),
            "n_cells": n_cells,
            "n_correct": n_correct,
            "tp": model_c["tp"], "fp": model_c["fp"],
            "fn": model_c["fn"], "tn": model_c["tn"],
            "static_correct": static_correct,
            "static_tp": static_c["tp"], "static_fp": static_c["fp"],
            "static_fn": static_c["fn"], "static_tn": static_c["tn"],
        })

    return {
        "version": 1,
        "grid_size": batch.height,
        "batch_size": batch.batch_size,
        "seed": batch.seed,
        "canonical_density": batch.canonical_density,
        "density_sweep": [float(d) for d in batch.densities],
        "sweep": sweep,
    }


def random_model_fn():
    """Return a `model_fn`-shaped callable emitting random logits.

    Signature matches a real model_fn exactly: takes (B, H, W) grids, returns a
    same-shaped float array. Pure NumPy, deterministic, no torch / GPU. Used by
    the smoke test:  task.evaluate(task.random_model_fn()).
    """
    def model_fn(grids: np.ndarray) -> np.ndarray:
        grids = np.asarray(grids)
        rng = np.random.default_rng(0)
        return rng.normal(0.0, 1.0, size=grids.shape).astype(np.float32)

    return model_fn
