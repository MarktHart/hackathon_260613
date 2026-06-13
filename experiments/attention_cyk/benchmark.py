"""Benchmark for attention_cyk.

Consumes the payload produced by ``task.evaluate`` and returns a flat dict of
scalar metrics describing how well a mechanism's per-cell attention lands on
the CYK-correct split points.

Headline:  ``cyk_split_accuracy_canonical`` -- cell-weighted probability mass
on correct split points across every query cell. Compare against
``uniform_baseline_accuracy`` (the same quantity under uniform attention).
"""

from __future__ import annotations

import math

VERSION = 1
GPU_REQUIREMENT = 1


def _num(x) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"expected a number, got {type(x).__name__}: {x!r}")
    return float(x)


def score(payload: dict) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )
    sweep = payload.get("sweep")
    if sweep is None:
        raise KeyError("payload missing required key 'sweep'")
    if not isinstance(sweep, list):
        raise ValueError("'sweep' must be a list of per-span records")

    metrics: dict[str, float | int] = {"version": VERSION}

    total_cells = 0
    acc_weighted = 0.0
    base_weighted = 0.0
    slice_accs: list[float] = []

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        for key in ("span_len", "num_cells", "split_accuracy", "uniform_baseline"):
            if key not in rec:
                raise KeyError(f"sweep record missing key '{key}'")
        span_len = int(rec["span_len"])
        cells = int(rec["num_cells"])
        acc = _num(rec["split_accuracy"])
        base = _num(rec["uniform_baseline"])

        metrics[f"split_accuracy_len_{span_len}"] = acc
        metrics[f"uniform_baseline_len_{span_len}"] = base

        total_cells += cells
        acc_weighted += acc * cells
        base_weighted += base * cells
        if cells > 0:
            slice_accs.append(acc)

    if total_cells > 0:
        canonical = acc_weighted / total_cells
        baseline = base_weighted / total_cells
    else:
        canonical = 0.0
        baseline = 0.0

    metrics["cyk_split_accuracy_canonical"] = canonical
    metrics["uniform_baseline_accuracy"] = baseline
    metrics["lift_over_uniform"] = canonical - baseline
    metrics["num_query_cells"] = int(total_cells)

    if slice_accs:
        mx = max(slice_accs)
        metrics["split_accuracy_robustness"] = (min(slice_accs) / mx) if mx > 0 else 0.0
    else:
        metrics["split_accuracy_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    canonical = metrics.get("cyk_split_accuracy_canonical")
    baseline = metrics.get("uniform_baseline_accuracy")
    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        # strictly worse than uniform attention -> degenerate, skip the jury
        if canonical < baseline:
            return True
    return False
