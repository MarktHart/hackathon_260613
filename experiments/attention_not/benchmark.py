"""Benchmark for the attention_not goal.

Pure Python. Deterministic. No I/O, no network, no imports from attempt dirs.

The payload (from task.evaluate) is a sweep over cos(theta) between the
attend-feature and suppress-feature directions, plus an identically-measured
linear baseline (a head with no NOT mechanism). We score how cleanly the head
separates the inhibited case (B present) from the active case (B absent), and
how robust that separation is as the two directions enter superposition.

Direction-of-better: every metric here is bigger-is-better.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1

# Must match task.CANONICAL_COS.
COS_SWEEP: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8]
CANONICAL_COS: float = 0.0  # orthogonal anchor

_REC_KEYS = ("not_sharpness", "suppression_gap", "attend_specificity")


def _fmt_cos(c: float) -> str:
    """0.0 -> '0p0', 0.6 -> '0p6'."""
    return f"{c:.1f}".replace(".", "p")


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate_records(records: Any, name: str) -> None:
    if not isinstance(records, list) or len(records) != len(COS_SWEEP):
        raise ValueError(
            f"payload['{name}'] must be a list of length {len(COS_SWEEP)}, "
            f"got {type(records).__name__} len "
            f"{len(records) if isinstance(records, list) else 'n/a'}"
        )
    for i, (expected_c, rec) in enumerate(zip(COS_SWEEP, records)):
        if not isinstance(rec, dict):
            raise ValueError(f"{name}[{i}] must be a dict")
        got = rec.get("cos")
        if not _num(got) or abs(float(got) - expected_c) > 1e-9:
            raise ValueError(f"{name}[{i}]['cos'] = {got!r}, expected {expected_c}")
        for k in _REC_KEYS:
            if k not in rec:
                raise KeyError(f"{name}[{i}] missing key {k!r}")
            v = rec[k]
            if not _num(v) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"{name}[{i}][{k!r}] must be finite, got {v!r}")


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    if "baseline" not in payload:
        raise KeyError("payload missing 'baseline'")
    _validate_records(payload["sweep"], "sweep")
    _validate_records(payload["baseline"], "baseline")

    sweep = payload["sweep"]
    baseline = payload["baseline"]

    sharp = {COS_SWEEP[i]: float(sweep[i]["not_sharpness"]) for i in range(len(COS_SWEEP))}
    base = {COS_SWEEP[i]: float(baseline[i]["not_sharpness"]) for i in range(len(COS_SWEEP))}

    metrics: dict[str, float | int] = {"version": VERSION}

    # ---- per-slice values + baseline + lift ----
    for i, c in enumerate(COS_SWEEP):
        key = _fmt_cos(c)
        metrics[f"not_sharpness_cos_{key}"] = sharp[c]
        metrics[f"linear_baseline_sharpness_cos_{key}"] = base[c]
        metrics[f"lift_over_baseline_cos_{key}"] = sharp[c] - base[c]
        metrics[f"suppression_gap_cos_{key}"] = float(sweep[i]["suppression_gap"])
        metrics[f"attend_specificity_cos_{key}"] = float(sweep[i]["attend_specificity"])

    # ---- canonical convenience metrics ----
    ck = _fmt_cos(CANONICAL_COS)
    metrics["not_sharpness_canonical"] = metrics[f"not_sharpness_cos_{ck}"]
    metrics["linear_baseline_sharpness_canonical"] = metrics[f"linear_baseline_sharpness_cos_{ck}"]
    metrics["lift_over_baseline_canonical"] = metrics[f"lift_over_baseline_cos_{ck}"]

    # ---- headline: superposition_robustness ----
    # Worst slice relative to canonical, clamped to [0, 1]. 1.0 => no
    # degradation as the attend/suppress features align. A head whose NOT
    # collapses under superposition scores near 0. Re-centred on chance (0.5)
    # so a head at chance everywhere scores 0, not 1.
    canon = sharp[CANONICAL_COS]
    if canon > 0.5:
        worst = min(sharp.values())
        robustness = (worst - 0.5) / (canon - 0.5)
        robustness = max(0.0, min(1.0, robustness))
    else:
        robustness = 0.0
    metrics["superposition_robustness"] = float(robustness)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical short-circuit before the (expensive) jury. Only fires on
    clearly degenerate attempts, never on borderline-but-real ones."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    sharp = metrics.get("not_sharpness_canonical")
    base = metrics.get("linear_baseline_sharpness_canonical")
    if _num(sharp) and _num(base):
        # A genuine NOT must clear the no-mechanism baseline / chance by a
        # margin at the easiest (orthogonal) condition.
        if sharp <= max(base, 0.5) + 0.1:
            return True
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
