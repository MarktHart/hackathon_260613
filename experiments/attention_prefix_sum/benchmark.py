"""Scoring for attention_prefix_sum."""

from __future__ import annotations

import math
from typing import Any

VERSION = 1


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """
    Compute metrics from payload.
    Payload must contain: version, sweep (list of {seq_len, correct, total}),
    random_baseline_correct, random_baseline_total.
    """
    # ---- Contract validation ----
    if "version" not in payload:
        raise KeyError("payload missing 'version'")
    if payload["version"] != VERSION:
        raise ValueError(f"payload version {payload['version']} != benchmark VERSION {VERSION}")

    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload 'sweep' must be a non-empty list")

    required_sweep_keys = {"seq_len", "correct", "total"}
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise TypeError(f"sweep[{i}] is not a dict")
        missing = required_sweep_keys - rec.keys()
        if missing:
            raise KeyError(f"sweep[{i}] missing keys: {missing}")
        if not isinstance(rec["correct"], int) or not isinstance(rec["total"], int):
            raise TypeError(f"sweep[{i}] correct/total must be int")

    for key in ("random_baseline_correct", "random_baseline_total"):
        if key not in payload:
            raise KeyError(f"payload missing '{key}'")
        if not isinstance(payload[key], int):
            raise TypeError(f"payload['{key}'] must be int")

    # ---- Compute per-length accuracies ----
    acc_by_len = {}
    for rec in sweep:
        L = rec["seq_len"]
        total = rec["total"]
        if total == 0:
            acc = 0.0
        else:
            acc = rec["correct"] / total
        acc_by_len[L] = acc

    canonical_len = 16
    if canonical_len not in acc_by_len:
        raise ValueError(f"sweep missing canonical seq_len={canonical_len}")

    prefix_acc_canonical = acc_by_len[canonical_len]

    # ---- Baseline ----
    rb_total = payload["random_baseline_total"]
    rb_correct = payload["random_baseline_correct"]
    linear_baseline_acc_canonical = rb_correct / rb_total if rb_total > 0 else 0.0

    # ---- Build metrics dict ----
    metrics: dict[str, float | int] = {
        "version": VERSION,
        "prefix_acc_canonical": prefix_acc_canonical,
        "prefix_acc_len_4": acc_by_len.get(4, 0.0),
        "prefix_acc_len_8": acc_by_len.get(8, 0.0),
        "prefix_acc_len_16": prefix_acc_canonical,
        "prefix_acc_len_32": acc_by_len.get(32, 0.0),
        "prefix_acc_len_64": acc_by_len.get(64, 0.0),
        "linear_baseline_acc_canonical": linear_baseline_acc_canonical,
        "lift_over_baseline_canonical": prefix_acc_canonical - linear_baseline_acc_canonical,
    }

    # Robustness: min accuracy across lengths / canonical accuracy
    all_accs = [acc_by_len.get(L, 0.0) for L in [4, 8, 16, 32, 64]]
    min_acc = min(all_accs)
    if prefix_acc_canonical > 0:
        metrics["prefix_acc_robustness"] = min_acc / prefix_acc_canonical
    else:
        metrics["prefix_acc_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """
    Detect clearly degenerate results.
    Returns True if:
    - any metric is NaN or inf
    - canonical accuracy <= baseline (no better than random)
    - robustness is NaN/inf
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    canonical = metrics.get("prefix_acc_canonical")
    baseline = metrics.get("linear_baseline_acc_canonical")
    robustness = metrics.get("prefix_acc_robustness")

    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        if canonical <= baseline:
            return True

    if isinstance(robustness, float) and (math.isnan(robustness) or math.isinf(robustness)):
        return True

    return False


# Pipeline knob: this is a lightweight CPU task, 1 GPU slot is fine
GPU_REQUIREMENT = 1