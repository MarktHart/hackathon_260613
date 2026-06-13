"""Benchmark for the `attention_xor` goal.

Pure Python, deterministic, side-effect free. Consumes the payload produced by
``task.evaluate`` and returns a flat dict of named scalars. See ``README.md``
for the payload contract and metric definitions.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1

# Marginal sweep values (must match task.SWEEP_PS, same ascending order).
SWEEP_PS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CANONICAL_P = 0.5

_REQUIRED_RECORD_KEYS = {"p", "accuracy", "baseline_accuracy", "frac_xor1", "n"}


def _fmt_p(p: float) -> str:
    # 0.1 -> 0p1, 0.5 -> 0p5, 1.0 -> 1p0
    return f"{p:.1f}".replace(".", "p")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _check_unit(name: str, v: Any) -> float:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        raise ValueError(f"{name} must be a number, got {v!r}")
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{name} must be finite, got {f}")
    return f


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    # ---- Input validation ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')} != benchmark VERSION {VERSION}"
        )
    sweep = payload.get("sweep")
    if not isinstance(sweep, list):
        raise KeyError("payload missing list 'sweep'")
    if len(sweep) != len(SWEEP_PS):
        raise ValueError(f"sweep must have length {len(SWEEP_PS)}, got {len(sweep)}")

    for i, (expected_p, rec) in enumerate(zip(SWEEP_PS, sweep)):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        missing = _REQUIRED_RECORD_KEYS - rec.keys()
        if missing:
            raise KeyError(f"sweep[{i}] missing keys: {sorted(missing)}")
        got_p = rec["p"]
        if not isinstance(got_p, (int, float)) or abs(float(got_p) - expected_p) > 1e-9:
            raise ValueError(f"sweep[{i}].p = {got_p!r}, expected {expected_p}")

    # ---- Build metrics ----
    metrics: dict[str, float | int] = {"version": VERSION}

    gap_captures: list[float] = []
    accuracies: list[float] = []
    for p, rec in zip(SWEEP_PS, sweep):
        key = _fmt_p(p)
        acc = _check_unit(f"sweep p={p} accuracy", rec["accuracy"])
        base = _check_unit(f"sweep p={p} baseline_accuracy", rec["baseline_accuracy"])

        gap = 1.0 - base
        # Baseline is the best-linear-probe floor; gap is how much headroom
        # remains above it. If there is no headroom (base == 1.0, only possible
        # degenerately), define full capture so the metric stays in [0, 1]
        # without dividing by 0.
        capture = 1.0 if gap <= 0.0 else _clamp01((acc - base) / gap)

        metrics[f"xor_accuracy_p_{key}"] = acc
        metrics[f"linear_baseline_accuracy_p_{key}"] = base
        metrics[f"lift_over_linear_p_{key}"] = acc - base
        metrics[f"xor_gap_capture_p_{key}"] = capture

        gap_captures.append(capture)
        accuracies.append(acc)

    # ---- Canonical convenience metrics ----
    canon = _fmt_p(CANONICAL_P)
    metrics["xor_accuracy_canonical"] = metrics[f"xor_accuracy_p_{canon}"]
    metrics["linear_baseline_accuracy_canonical"] = metrics[
        f"linear_baseline_accuracy_p_{canon}"
    ]
    metrics["lift_over_linear_canonical"] = metrics[f"lift_over_linear_p_{canon}"]

    # ---- Aggregates ----
    metrics["worst_slice_accuracy"] = min(accuracies) if accuracies else 0.0

    # Headline: mean fraction of the above-majority gap captured, in [0, 1].
    metrics["xor_robustness"] = (
        sum(gap_captures) / len(gap_captures) if gap_captures else 0.0
    )

    return metrics


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """Pipeline hook: True iff the attempt is mechanically degenerate.

    Catches NaN/inf and the case where the canonical condition does not beat the
    best-linear-probe baseline — i.e. no XOR was learned. Never returns True for
    a borderline-but-real result; it only ever short-circuits the jury.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    acc = metrics.get("xor_accuracy_canonical")
    base = metrics.get("linear_baseline_accuracy_canonical")
    if not isinstance(acc, (int, float)) or not isinstance(base, (int, float)):
        return True
    # At p=0.5 the best linear probe is ~0.75 (the majority floor is only ~0.5);
    # require a clear margin above the best-linear-probe baseline.
    if acc <= base + 0.05:
        return True

    robustness = metrics.get("xor_robustness")
    if not isinstance(robustness, (int, float)) or robustness <= 0.0:
        return True

    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
