"""Benchmark for the attention_palindrome goal.

Pure Python (stdlib only). Deterministic. No I/O, no network, no NumPy, no
imports from any attempt directory.

The payload (produced by task.evaluate) is a difficulty sweep over the number
of broken mirror pairs `k` in the negatives, plus an identically-measured
linear-on-histogram baseline. We score how well the model separates perfect
palindromes from the broken-pair negatives at each difficulty, and how robust
that separation is as the task gets subtle (fewer broken pairs).
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1

# Must match task.py.
MISMATCH_SWEEP: tuple[int, ...] = (1, 2, 4, 8)
CANONICAL_MISMATCH: int = 1  # the diagnostic anchor: a single broken pair


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _skill(auc: float) -> float:
    """Map AUC in [0, 1] to a skill in [0, 1]: chance (0.5) -> 0, perfect -> 1.
    Anti-correlation (auc < 0.5) is clamped to 0."""
    return max(0.0, 2.0 * (auc - 0.5))


def _validate_records(records: Any, name: str) -> None:
    if not isinstance(records, list) or len(records) != len(MISMATCH_SWEEP):
        raise ValueError(
            f"payload['{name}'] must be a list of length {len(MISMATCH_SWEEP)}"
        )
    for i, (expected_k, rec) in enumerate(zip(MISMATCH_SWEEP, records)):
        if not isinstance(rec, dict):
            raise ValueError(f"{name}[{i}] must be a dict")
        got = rec.get("mismatch")
        if not _num(got) or int(got) != expected_k:
            raise ValueError(
                f"{name}[{i}]['mismatch'] = {got!r}, expected {expected_k}"
            )
        auc = rec.get("auc")
        if auc is None:
            raise KeyError(f"{name}[{i}] missing key 'auc'")
        if not _num(auc) or math.isnan(auc) or math.isinf(auc):
            raise ValueError(f"{name}[{i}]['auc'] must be a finite number, got {auc!r}")
        if not (-1e-9 <= auc <= 1.0 + 1e-9):
            raise ValueError(f"{name}[{i}]['auc'] = {auc} out of [0, 1]")


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    # ---- validation ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    if "linear_baseline" not in payload:
        raise KeyError("payload missing 'linear_baseline'")
    _validate_records(payload["sweep"], "sweep")
    _validate_records(payload["linear_baseline"], "linear_baseline")

    sweep = payload["sweep"]
    baseline = payload["linear_baseline"]

    auc_by_k = {int(MISMATCH_SWEEP[i]): float(sweep[i]["auc"]) for i in range(len(MISMATCH_SWEEP))}
    base_by_k = {int(MISMATCH_SWEEP[i]): float(baseline[i]["auc"]) for i in range(len(MISMATCH_SWEEP))}

    metrics: dict[str, float | int] = {"version": VERSION}

    # ---- per-slice values + baselines + lift ----
    for k in MISMATCH_SWEEP:
        kk = int(k)
        metrics[f"auc_mismatch_k{kk}"] = auc_by_k[kk]
        metrics[f"linear_baseline_auc_mismatch_k{kk}"] = base_by_k[kk]
        metrics[f"lift_over_baseline_mismatch_k{kk}"] = auc_by_k[kk] - base_by_k[kk]

    # ---- canonical convenience metrics (k = 1, the hardest / most diagnostic) ----
    ck = int(CANONICAL_MISMATCH)
    metrics["auc_canonical"] = auc_by_k[ck]
    metrics["linear_baseline_auc_canonical"] = base_by_k[ck]
    metrics["lift_over_baseline_canonical"] = auc_by_k[ck] - base_by_k[ck]
    metrics["palindrome_skill_canonical"] = _skill(auc_by_k[ck])

    # ---- headline: palindrome_robustness ----
    # Ratio of the hardest-slice skill to the easiest-slice skill across the
    # sweep, in [0, 1]. 1.0 => detection holds up even when a single pair is
    # broken; ~0 => the model only catches grossly-broken negatives (a shortcut).
    skills = {int(k): _skill(auc_by_k[int(k)]) for k in MISMATCH_SWEEP}
    best_skill = max(skills.values())
    worst_skill = min(skills.values())
    if best_skill <= 1e-9:
        robustness = 0.0
    else:
        robustness = max(0.0, min(1.0, worst_skill / best_skill))
    metrics["palindrome_robustness"] = float(robustness)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical short-circuit before the (expensive) jury. Never True for a
    borderline-but-real result — only for clearly degenerate ones."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # A real palindrome mechanism must clear the histogram-readout floor at the
    # canonical (single broken pair) anchor by a margin.
    auc = metrics.get("auc_canonical")
    base = metrics.get("linear_baseline_auc_canonical")
    if _num(auc) and _num(base):
        if _skill(auc) <= _skill(base) + 0.05:
            return True
    return False


GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
