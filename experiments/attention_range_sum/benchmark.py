"""Benchmark for the attention_range_sum goal.

Pure Python. Deterministic. No I/O, no network, no imports from attempt dirs.

The payload (produced by task.evaluate) is a sweep over the range length `k`.
Each sweep record carries the model's flattened predictions and the matching
ground-truth range sums. We score mean-squared error per slice, compare it to
the no-mechanism constant-predictor baseline measured on the same targets, and
summarise how robustly the model holds up as the range grows.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1

# Must match task.py / README.
RANGE_LENS: list[int] = [2, 4, 8, 16, 32]
CANONICAL_RANGE_LEN: int = 8


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _mse(preds: list[float], targets: list[float]) -> float:
    if len(preds) != len(targets):
        raise ValueError(
            f"predictions/targets length mismatch: {len(preds)} vs {len(targets)}"
        )
    n = len(targets)
    if n == 0:
        return 0.0
    total = 0.0
    for p, t in zip(preds, targets):
        d = float(p) - float(t)
        total += d * d
    return total / n


def _variance(targets: list[float]) -> float:
    """MSE of the optimal constant predictor (per-slice mean of targets).

    This is the no-mechanism baseline: a model that knows nothing about the
    range but predicts the best single constant. MSE equals the variance of
    the targets. Computable from the payload's targets alone."""
    n = len(targets)
    if n == 0:
        return 0.0
    mean = sum(float(t) for t in targets) / n
    total = 0.0
    for t in targets:
        d = float(t) - mean
        total += d * d
    return total / n


def _check_floats(seq: Any, label: str) -> list[float]:
    if not isinstance(seq, list):
        raise ValueError(f"{label} must be a list, got {type(seq).__name__}")
    out: list[float] = []
    for i, v in enumerate(seq):
        if not _num(v) or math.isnan(v) or math.isinf(v):
            raise ValueError(f"{label}[{i}] must be a finite number, got {v!r}")
        out.append(float(v))
    return out


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    # ---- validation ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )
    sweep = payload.get("sweep")
    if not isinstance(sweep, list) or len(sweep) != len(RANGE_LENS):
        raise ValueError(
            f"payload['sweep'] must be a list of length {len(RANGE_LENS)}, "
            f"got {sweep!r}"
        )

    mse_by_k: dict[int, float] = {}
    base_by_k: dict[int, float] = {}

    for i, (expected_k, rec) in enumerate(zip(RANGE_LENS, sweep)):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        k = rec.get("range_len")
        if not isinstance(k, int) or k != expected_k:
            raise ValueError(
                f"sweep[{i}]['range_len'] = {k!r}, expected {expected_k}"
            )
        if "predictions" not in rec:
            raise KeyError(f"sweep[{i}] missing 'predictions'")
        if "targets" not in rec:
            raise KeyError(f"sweep[{i}] missing 'targets'")
        preds = _check_floats(rec["predictions"], f"sweep[{i}]['predictions']")
        targets = _check_floats(rec["targets"], f"sweep[{i}]['targets']")
        mse_by_k[k] = _mse(preds, targets)
        base_by_k[k] = _variance(targets)

    metrics: dict[str, float | int] = {"version": VERSION}

    # ---- per-slice values + baselines ----
    for k in RANGE_LENS:
        metrics[f"range_sum_mse_k_{k}"] = float(mse_by_k[k])
        metrics[f"linear_baseline_mse_k_{k}"] = float(base_by_k[k])

    # ---- canonical convenience metric ----
    metrics["range_sum_mse_canonical"] = float(mse_by_k[CANONICAL_RANGE_LEN])

    # ---- lift over baseline at canonical (baseline - model; larger better) ----
    metrics["lift_over_linear_k_8"] = float(
        base_by_k[CANONICAL_RANGE_LEN] - mse_by_k[CANONICAL_RANGE_LEN]
    )

    # ---- headline: robustness across the sweep ----
    # Ratio of canonical MSE to the hardest-slice MSE. 1.0 => no degradation as
    # the range grows. Clamped to [0, 1]; zero denominator (a perfect hardest
    # slice) yields 1.0.
    canon = mse_by_k[CANONICAL_RANGE_LEN]
    hardest = mse_by_k[RANGE_LENS[-1]]
    if hardest <= 1e-12:
        robustness = 1.0
    else:
        robustness = canon / hardest
    metrics["range_sum_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical short-circuit before the (expensive) jury. Never True for a
    borderline-but-real result — only for clearly degenerate ones."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    mse = metrics.get("range_sum_mse_canonical")
    baseline = metrics.get("linear_baseline_mse_k_8")
    if _num(mse) and _num(baseline):
        # A genuine range-sum head must beat the constant-predictor floor by a
        # clear margin (>= 10% reduction in MSE at the canonical range).
        if mse >= baseline * 0.9:
            return True
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
