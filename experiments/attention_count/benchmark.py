import math
from typing import Any

VERSION = 1

def score(payload: dict[str, Any]) -> dict[str, float | int]:
    # Validate payload contract
    required_keys = [
        "version", "n_layers", "n_heads", "ground_truth_induction_heads",
        "per_head_scores", "threshold_sweep"
    ]
    for key in required_keys:
        if key not in payload:
            raise KeyError(f"payload missing '{key}'")
    
    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version: {payload['version']}")
    
    gt = payload["ground_truth_induction_heads"]
    per_head_scores = payload["per_head_scores"]
    threshold_sweep = payload["threshold_sweep"]
    
    # Validate lengths
    expected_heads = payload["n_layers"] * payload["n_heads"]
    if len(per_head_scores) != expected_heads:
        raise ValueError(f"per_head_scores length {len(per_head_scores)} != n_layers*n_heads {expected_heads}")
    if len(threshold_sweep) != 21:
        raise ValueError(f"threshold_sweep length {len(threshold_sweep)} != 21")
    
    # Compute accuracy at each threshold
    accuracies = {}
    for entry in threshold_sweep:
        thr = entry["threshold"]
        pred = entry["predicted_count"]
        # Accuracy: 1 - |pred - gt| / gt  (gt=2, so max error 2 -> min accuracy 0)
        # But if pred > 2*gt, clip at 0
        err = abs(pred - gt)
        acc = max(0.0, 1.0 - err / gt)
        key = f"count_accuracy_thr_{thr:.2f}".replace(".", "p")
        accuracies[key] = acc
    
    # Headline: accuracy at threshold 0.5
    headline = accuracies.get("count_accuracy_thr_0p50", 0.0)
    
    # AUC over threshold sweep (trapezoidal rule)
    # thresholds are uniformly spaced 0.00..1.00 step 0.05
    auc = 0.0
    for i in range(len(threshold_sweep) - 1):
        t1 = threshold_sweep[i]["threshold"]
        t2 = threshold_sweep[i + 1]["threshold"]
        a1 = accuracies[f"count_accuracy_thr_{t1:.2f}".replace(".", "p")]
        a2 = accuracies[f"count_accuracy_thr_{t2:.2f}".replace(".", "p")]
        auc += 0.5 * (a1 + a2) * (t2 - t1)
    
    # Baseline: always guess midpoint (4 heads)
    baseline_pred = expected_heads // 2  # 4
    baseline_err = abs(baseline_pred - gt)
    baseline_acc = max(0.0, 1.0 - baseline_err / gt)
    
    metrics = {
        "version": 1,
        "count_accuracy_canonical": headline,
        "auc_count": auc,
        "baseline_accuracy_canonical": baseline_acc,
        "lift_over_baseline": headline - baseline_acc,
    }
    # Add per-threshold metrics
    metrics.update(accuracies)
    
    return metrics

def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    # Fail fast on NaN/inf
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # If headline accuracy is worse than baseline, it's broken
    headline = metrics.get("count_accuracy_canonical", 0.0)
    baseline = metrics.get("baseline_accuracy_canonical", 0.0)
    if headline <= baseline:
        return True
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
