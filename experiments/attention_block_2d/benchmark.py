"""Scoring for attention_block_2d."""

from __future__ import annotations

import math
from typing import Any

VERSION = 1


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """
    Compute metrics from a payload produced by task.evaluate().
    Returns flat dict of scalar metrics.
    """
    # ---- Contract validation ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(f"payload version {payload.get('version')} != benchmark VERSION {VERSION}")
    if "grid_size" not in payload:
        raise KeyError("payload missing 'grid_size'")
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    # ---- Basic counts ----
    n_total = len(sweep)
    n_correct = sum(1 for r in sweep if r.get("correct", False))
    pattern_acc_canonical = n_correct / n_total

    # ---- Per-pattern accuracies ----
    pattern_ids = ["local", "dilated", "global", "causal_2d"]
    per_pattern = {}
    for pid in pattern_ids:
        subset = [r for r in sweep if r["pattern_id"] == pid]
        if subset:
            acc = sum(1 for r in subset if r.get("correct", False)) / len(subset)
        else:
            acc = 0.0
        per_pattern[f"pattern_acc_{pid}"] = acc

    # ---- Confidence statistics ----
    correct_conf = [r["confidence"] for r in sweep if r.get("correct", False)]
    incorrect_conf = [r["confidence"] for r in sweep if not r.get("correct", False)]
    mean_conf_correct = float(np_mean(correct_conf)) if correct_conf else 0.0
    mean_conf_incorrect = float(np_mean(incorrect_conf)) if incorrect_conf else 0.0

    # ---- Linear baseline (always predicts "local") ----
    # Baseline accuracy = fraction of examples that are actually "local"
    n_local = sum(1 for r in sweep if r["pattern_id"] == "local")
    linear_baseline_acc = n_local / n_total

    # ---- Summary metrics ----
    lift = pattern_acc_canonical - linear_baseline_acc

    return {
        "version": VERSION,
        "pattern_acc_canonical": pattern_acc_canonical,
        "pattern_acc_local": per_pattern["pattern_acc_local"],
        "pattern_acc_dilated": per_pattern["pattern_acc_dilated"],
        "pattern_acc_global": per_pattern["pattern_acc_global"],
        "pattern_acc_causal_2d": per_pattern["pattern_acc_causal_2d"],
        "mean_confidence_correct": mean_conf_correct,
        "mean_confidence_incorrect": mean_conf_incorrect,
        "linear_baseline_acc": linear_baseline_acc,
        "lift_over_linear_baseline": lift,
    }


def np_mean(vals: list[float]) -> float:
    """Pure-Python mean to avoid numpy dependency in benchmark."""
    return sum(vals) / len(vals)


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """
    Pipeline hook: return True iff metrics indicate a catastrophic failure
    (NaN, inf, or accuracy not beating the trivial baseline by a meaningful margin).
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    acc = metrics.get("pattern_acc_canonical")
    baseline = metrics.get("linear_baseline_acc")
    if isinstance(acc, (int, float)) and isinstance(baseline, (int, float)):
        # Require at least 1.5× baseline (i.e. > 3/16 = 0.1875 for canonical 4/16 baseline)
        if acc <= baseline * 1.5:
            return True

    return False