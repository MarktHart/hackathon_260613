import math
from typing import Dict, Any

VERSION = 1

def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    if payload.get("version") != VERSION:
        raise ValueError(f"Payload version {payload.get('version')} != benchmark VERSION {VERSION}")

    required_keys = ["canonical", "sweep", "config", "baseline"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing payload key: {k}")

    # Canonical metrics
    pred = np.array(payload["canonical"]["pred"])
    target = np.array(payload["canonical"]["target"])
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
    mse_canon = float(np.mean((pred - target) ** 2))
    var_target = float(np.var(target))
    r2_canon = 1.0 - mse_canon / var_target if var_target > 0 else 0.0

    # Baseline metrics
    baseline = payload["baseline"]
    baseline_mse = float(baseline["mse_canonical"])
    baseline_r2 = float(baseline["r2_canonical"])

    # Per-slice metrics
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Sweep must be non-empty list")
    r2_vals = []
    metrics = {
        "version": VERSION,
        "linear_combination_r2_canonical": r2_canon,
        "linear_combination_mse_canonical": mse_canon,
        "linear_baseline_r2_canonical": baseline_r2,
        "linear_baseline_mse_canonical": baseline_mse,
        "lift_over_baseline_r2": r2_canon - baseline_r2,
    }
    for rec in sweep:
        a = rec["alpha"]
        b = rec["beta"]
        r2 = float(rec["r2"])
        mse = float(rec["mse"])
        # format floats as 1p0, m1p0, 2p0, m2p0
        def fmt(v: float) -> str:
            s = f"{v:.1f}".replace(".", "p").replace("-", "m")
            return s
        a_str = fmt(a)
        b_str = fmt(b)
        metrics[f"linear_combination_r2_alpha_{a_str}_beta_{b_str}"] = r2
        metrics[f"linear_combination_mse_alpha_{a_str}_beta_{b_str}"] = mse
        r2_vals.append(r2)

    # Robustness: min/max R² across sweep
    if r2_vals:
        r2_min = min(r2_vals)
        r2_max = max(r2_vals)
        robustness = r2_min / r2_max if r2_max > 0 else 0.0
        metrics["linear_combination_robustness"] = float(robustness)
    else:
        metrics["linear_combination_robustness"] = 0.0

    return metrics

def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    r2 = metrics.get("linear_combination_r2_canonical")
    baseline = metrics.get("linear_baseline_r2_canonical")
    if isinstance(r2, (int, float)) and isinstance(baseline, (int, float)):
        if r2 <= baseline * 1.01:  # not meaningfully better than baseline
            return True
    rob = metrics.get("linear_combination_robustness")
    if isinstance(rob, float) and rob < 0.1:  # collapses on some coefficients
        return True
    return False

# Need numpy for score()
import numpy as np

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
