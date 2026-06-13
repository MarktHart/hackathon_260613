import math
from typing import Dict, Any

import numpy as np

VERSION = 1

def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    _validate_payload(payload)
    metrics = {"version": VERSION}
    # Canonical headline metrics
    metrics["dyck_matching_canonical"] = payload["aggregated"]["best_matching_accuracy"]
    metrics["dyck_depth_corr_canonical"] = payload["aggregated"]["best_depth_corr"]
    # Per-head slices
    for ph in payload["per_head"]:
        h = ph["head"]
        metrics[f"dyck_head_{h}_matching"] = ph["matching_accuracy"]
        metrics[f"dyck_head_{h}_depth_corr"] = ph["depth_corr"]
    # Diagnostic: mean diagonal fraction
    diag_fracs = [ph["diag_frac"] for ph in payload["per_head"]]
    metrics["dyck_diag_frac_mean"] = float(np.mean(diag_fracs)) if diag_fracs else 0.0
    # Linear baseline: a fixed head attending uniformly to all prior open brackets.
    # Computed deterministically on the canonical batch inside task.evaluate and
    # carried in the payload, so it tracks the canonical seed/scale rather than
    # being a stale hardcoded constant.
    linear_baseline = payload["aggregated"]["linear_baseline_matching"]
    metrics["linear_baseline_matching"] = linear_baseline
    metrics["lift_over_baseline_matching"] = metrics["dyck_matching_canonical"] - linear_baseline
    return metrics

def _validate_payload(payload: Dict[str, Any]) -> None:
    required = ["version", "canonical_seed", "seq_len", "max_depth", "n_heads", "n_layers", "per_head", "aggregated"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing payload key: {k}")
    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version: {payload['version']}")
    if not isinstance(payload["per_head"], list) or len(payload["per_head"]) != payload["n_heads"]:
        raise ValueError("per_head length mismatch")
    for ph in payload["per_head"]:
        for k in ["head", "matching_accuracy", "depth_corr", "diag_frac"]:
            if k not in ph:
                raise KeyError(f"Missing per_head key: {k}")
    for k in ["best_matching_accuracy", "best_depth_corr", "linear_baseline_matching"]:
        if k not in payload["aggregated"]:
            raise KeyError(f"Missing aggregated key: {k}")

def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    sharp = metrics.get("dyck_matching_canonical")
    baseline = metrics.get("linear_baseline_matching")
    if isinstance(sharp, (int, float)) and isinstance(baseline, (int, float)):
        if sharp <= baseline * 1.1:  # barely beats uniform baseline
            return True
    return False