"""Benchmark for the `attention_game_of_life` goal.

Consumes the payload from `task.evaluate` (a sweep over initial board density)
and returns flat scalar metrics: a headline F1 at the canonical density,
per-slice values, and a static-copy baseline measured under identical boards.

Pure Python. Deterministic. No I/O, no imports from any attempt directory.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs a model on the GPU.
GPU_REQUIREMENT = 1


def _density_key(prefix: str, density: float) -> str:
    """Slice key, e.g. ('next_state_f1', 0.3) -> 'next_state_f1_density_0p3'."""
    return f"{prefix}_density_{density:.1f}".replace(".", "p")


def _f1(tp: int, fp: int, fn: int) -> float:
    """F1 of the positive ('alive next') class, 0.0 when undefined."""
    denom = 2 * tp + fp + fn
    if denom <= 0:
        return 0.0
    return (2.0 * tp) / denom


def _recall(tp: int, fn: int) -> float:
    if tp + fn <= 0:
        return 0.0
    return tp / (tp + fn)


def _precision(tp: int, fp: int) -> float:
    if tp + fp <= 0:
        return 0.0
    return tp / (tp + fp)


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), canonical_density (float),
        density_sweep (list[float]), sweep (list[record]).

    Each sweep record:
        {density, n_cells, n_correct,
         tp, fp, fn, tn,
         static_correct, static_tp, static_fp, static_fn, static_tn}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_density", "density_sweep", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    density_sweep = payload["density_sweep"]
    if not isinstance(density_sweep, (list, tuple)) or len(density_sweep) == 0:
        raise ValueError("payload['density_sweep'] must be a non-empty list")
    density_sweep = [float(d) for d in density_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(density_sweep):
        raise ValueError(
            "payload['sweep'] must be a list of the same length as density_sweep"
        )

    # Index records by nominal density.
    by_density: dict = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        if "density" not in rec:
            raise KeyError("sweep record missing 'density'")
        by_density[round(float(rec["density"]), 6)] = rec

    metrics: dict[str, float | int] = {"version": VERSION}

    f1_vals: list[float] = []
    acc_vals: list[float] = []
    static_f1_vals: list[float] = []

    for density in density_sweep:
        rec = by_density.get(round(density, 6), {})

        n_cells = int(rec.get("n_cells", 0))
        n_correct = int(rec.get("n_correct", 0))
        tp = int(rec.get("tp", 0))
        fp = int(rec.get("fp", 0))
        fn = int(rec.get("fn", 0))

        static_correct = int(rec.get("static_correct", 0))
        s_tp = int(rec.get("static_tp", 0))
        s_fp = int(rec.get("static_fp", 0))
        s_fn = int(rec.get("static_fn", 0))

        accuracy = (n_correct / n_cells) if n_cells > 0 else 0.0
        f1 = _f1(tp, fp, fn)
        recall = _recall(tp, fn)
        precision = _precision(tp, fp)

        static_accuracy = (static_correct / n_cells) if n_cells > 0 else 0.0
        static_f1 = _f1(s_tp, s_fp, s_fn)

        metrics[_density_key("next_state_accuracy", density)] = float(accuracy)
        metrics[_density_key("next_state_f1", density)] = float(f1)
        metrics[_density_key("live_recall", density)] = float(recall)
        metrics[_density_key("live_precision", density)] = float(precision)
        metrics[_density_key("static_baseline_accuracy", density)] = float(static_accuracy)
        metrics[_density_key("static_baseline_f1", density)] = float(static_f1)

        f1_vals.append(float(f1))
        acc_vals.append(float(accuracy))
        static_f1_vals.append(float(static_f1))

    # --- Canonical condition ---
    canonical = float(payload["canonical_density"])
    canonical_f1 = float(metrics.get(_density_key("next_state_f1", canonical), 0.0))
    canonical_acc = float(metrics.get(_density_key("next_state_accuracy", canonical), 0.0))
    canonical_static_f1 = float(
        metrics.get(_density_key("static_baseline_f1", canonical), 0.0)
    )

    metrics["next_state_f1_canonical"] = canonical_f1
    metrics["next_state_accuracy_canonical"] = canonical_acc
    metrics["lift_over_static_f1_canonical"] = canonical_f1 - canonical_static_f1

    # --- Aggregates ---
    n = len(f1_vals)
    metrics["mean_next_state_f1"] = float(sum(f1_vals) / n) if n else 0.0
    metrics["mean_next_state_accuracy"] = float(sum(acc_vals) / n) if n else 0.0

    # --- Headline: life_robustness ---
    # Worst-case next-state F1 across the density sweep. Bigger is better; in
    # [0, 1]. A method that only works at one density scores poorly here.
    metrics["life_robustness"] = float(min(f1_vals)) if f1_vals else 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Pipeline hook: True if metrics are clearly degenerate, to skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that cannot beat the trivial 'copy the current board' baseline
    # at the canonical density has learned no Game of Life dynamics.
    f1 = metrics.get("next_state_f1_canonical")
    static_key = "static_baseline_f1_density_0p3"
    static = metrics.get(static_key)
    if isinstance(f1, (int, float)) and isinstance(static, (int, float)):
        if f1 <= static:
            return True

    return False
