"""Benchmark for the `attention_bfs` goal.

Consumes the payload produced by ``task.evaluate`` and returns a flat dict of
named scalar metrics. Pure Python, deterministic, side-effect free.

Headline metric
---------------
``bfs_f1_canonical`` — pooled F1 of the attempt's reachability prediction at the
canonical hop budget (the full BFS horizon). Bigger is better. This is the one
number an attempt should optimise.
"""

from __future__ import annotations

import math

VERSION = 1

# Pipeline hook: this goal is GPU-bound for attempts (they run a real model),
# but task/benchmark stay pure CPU. The minimum slot count is 1.
GPU_REQUIREMENT = 1


def _fmt(x: float | int) -> str:
    """Format a hop value for a metric key (ints stay ints; floats use 0p7)."""
    if isinstance(x, int) or float(x).is_integer():
        return str(int(x))
    return ("%g" % x).replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    # ---- Contract validation ----
    for key in ("version", "canonical_hops", "sweep"):
        if key not in payload:
            raise KeyError(f"Payload missing required key: {key!r}")

    if payload["version"] != VERSION:
        raise ValueError(
            f"Payload version {payload['version']} != benchmark VERSION {VERSION}"
        )

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Payload 'sweep' must be a non-empty list")

    canonical_hops = int(payload["canonical_hops"])

    by_hops: dict[int, dict] = {}
    for rec in sweep:
        for f in ("hops", "model_f1", "model_acc", "baseline_f1"):
            if f not in rec:
                raise KeyError(f"Sweep record missing required field: {f!r}")
        by_hops[int(rec["hops"])] = rec

    metrics: dict[str, float | int] = {"version": VERSION}

    # ---- Per-slice metrics ----
    model_f1s = []
    for h in sorted(by_hops):
        rec = by_hops[h]
        tag = _fmt(h)
        metrics[f"bfs_f1_hops_{tag}"] = float(rec["model_f1"])
        metrics[f"bfs_acc_hops_{tag}"] = float(rec["model_acc"])
        metrics[f"linear_baseline_f1_hops_{tag}"] = float(rec["baseline_f1"])
        model_f1s.append(float(rec["model_f1"]))

    # ---- Canonical (headline) ----
    if canonical_hops not in by_hops:
        raise ValueError(
            f"canonical_hops={canonical_hops} not present in sweep "
            f"(have {sorted(by_hops)})"
        )
    canon = by_hops[canonical_hops]
    metrics["bfs_f1_canonical"] = float(canon["model_f1"])
    metrics["bfs_acc_canonical"] = float(canon["model_acc"])
    metrics["linear_baseline_f1_canonical"] = float(canon["baseline_f1"])
    metrics["lift_over_linear_baseline"] = float(
        canon["model_f1"] - canon["baseline_f1"]
    )

    # ---- Aggregate over the whole sweep ----
    metrics["bfs_f1_mean"] = sum(model_f1s) / len(model_f1s)

    # Robustness: how well F1 holds as the hop budget grows (deep propagation)
    # relative to the easiest 1-hop case. Clamped to [0, 1].
    min_h = min(by_hops)
    max_h = max(by_hops)
    f1_easy = float(by_hops[min_h]["model_f1"])
    f1_hard = float(by_hops[max_h]["model_f1"])
    if f1_easy <= 0.0:
        robustness = 0.0
    else:
        robustness = max(0.0, min(1.0, f1_hard / f1_easy))
    metrics["bfs_reachability_robustness"] = robustness

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Short-circuit the jury for mechanically-detectable failures."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    f1 = metrics.get("bfs_f1_canonical")
    if isinstance(f1, int | float) and not (0.0 <= f1 <= 1.0):
        return True

    baseline = metrics.get("linear_baseline_f1_canonical")
    # The whole point is multi-hop propagation beyond the 1-hop baseline.
    # A method that does not beat the baseline at the canonical horizon adds
    # nothing mechanistically interesting.
    if isinstance(f1, int | float) and isinstance(baseline, int | float):
        if f1 <= baseline:
            return True

    return False
