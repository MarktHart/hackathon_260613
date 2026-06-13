import math
from typing import Dict, List, Any

VERSION = 1


def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    # Validate payload structure
    required_keys = {"version", "canonical_threshold", "sweep", "n_heads", "seq_len"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Payload missing required keys: {missing}")

    if payload["version"] != VERSION:
        raise ValueError(f"Payload version {payload['version']} != benchmark VERSION {VERSION}")

    sweep: List[Dict[str, Any]] = payload["sweep"]
    if not sweep:
        raise ValueError("Payload sweep is empty")

    # Validate sweep records
    for i, rec in enumerate(sweep):
        for k in ("run_length", "difficulty", "mae", "rmse", "correlation", "n_samples"):
            if k not in rec:
                raise KeyError(f"Sweep record {i} missing key: {k}")

    # --- Helper: average over run lengths for a given difficulty ---
    def avg_over_L(difficulty: float, metric: str) -> float:
        vals = [r[metric] for r in sweep if abs(r["difficulty"] - difficulty) < 1e-6]
        if not vals:
            return float("nan")
        return float(np_mean(vals))

    # --- Helper: average over difficulties for a given run length ---
    def avg_over_d(run_length: int, metric: str) -> float:
        vals = [r[metric] for r in sweep if r["run_length"] == run_length]
        if not vals:
            return float("nan")
        return float(np_mean(vals))

    # --- Canonical difficulty is d=0.5 ---
    canonical_d = 0.5
    mae_canonical = avg_over_L(canonical_d, "mae")
    rmse_canonical = avg_over_L(canonical_d, "rmse")
    corr_canonical = avg_over_L(canonical_d, "correlation")

    # --- Per-difficulty metrics ---
    difficulties = sorted({r["difficulty"] for r in sweep})
    metrics = {
        "version": VERSION,
        "longest_run_mae_canonical": mae_canonical,
        "longest_run_rmse_canonical": rmse_canonical,
        "longest_run_corr_canonical": corr_canonical,
    }

    for d in difficulties:
        d_str = f"{d:.1f}".replace(".", "p")
        metrics[f"longest_run_mae_d_{d_str}"] = avg_over_L(d, "mae")
        metrics[f"longest_run_rmse_d_{d_str}"] = avg_over_L(d, "rmse")
        metrics[f"longest_run_corr_d_{d_str}"] = avg_over_L(d, "correlation")

    # --- Per-run-length metrics ---
    run_lengths = sorted({r["run_length"] for r in sweep})
    for L in run_lengths:
        metrics[f"longest_run_mae_L_{L}"] = avg_over_d(L, "mae")

    # --- Linear baseline (always predict mean run length) ---
    # Mean run length in the canonical sweep (d=0.5) = average of the run_lengths
    # since each run length has equal samples
    mean_run_length = float(np_mean([r["run_length"] for r in sweep if abs(r["difficulty"] - canonical_d) < 1e-6]))
    # Baseline MAE = mean absolute deviation from mean_run_length
    baseline_mae_vals = []
    for r in sweep:
        if abs(r["difficulty"] - canonical_d) < 1e-6:
            L = r["run_length"]
            # True value is L, prediction is mean_run_length
            baseline_mae_vals.append(abs(L - mean_run_length))
    linear_baseline_mae = float(np_mean(baseline_mae_vals)) if baseline_mae_vals else float("nan")
    metrics["linear_baseline_mae_canonical"] = linear_baseline_mae

    # --- Lift over baseline ---
    if not (math.isnan(mae_canonical) or math.isnan(linear_baseline_mae)):
        metrics["lift_over_linear_baseline_mae"] = linear_baseline_mae - mae_canonical
    else:
        metrics["lift_over_linear_baseline_mae"] = float("nan")

    return metrics


def np_mean(vals):
    """Pure Python mean to avoid numpy dependency in benchmark."""
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    # NaN/inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Performance worse than or equal to linear baseline (no lift)
    mae = metrics.get("longest_run_mae_canonical")
    baseline = metrics.get("linear_baseline_mae_canonical")
    if isinstance(mae, (int, float)) and isinstance(baseline, (int, float)):
        # MAE should be *smaller* than baseline; if not, method is not learning
        if mae >= baseline * 1.0:  # >= because smaller is better
            return True

    # Correlation should be positive at canonical difficulty
    corr = metrics.get("longest_run_corr_canonical")
    if isinstance(corr, (int, float)) and corr <= 0.0:
        return True

    return False


# Optional: this goal is lightweight, 1 GPU is plenty
GPU_REQUIREMENT = 1