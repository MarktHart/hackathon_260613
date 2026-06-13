import math

VERSION = 1


def _safe_div(num: float, den: float) -> float:
    """Return num/den if den > 0 else 0.0. Never returns inf or NaN."""
    if den <= 0.0:
        return 0.0
    return num / den


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute all metrics from the payload produced by task.evaluate().
    Returns a flat dict of scalar metrics.
    """
    # --- validate payload contract ---
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported payload version: {payload.get('version')}")
    sweep = payload.get("sweep")
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    required_keys = {
        "cos", "s_A_at_A", "s_A_at_B", "s_A_noise_max",
        "s_B_at_A", "s_B_at_B", "s_B_noise_max",
        "s_AB_at_A", "s_AB_at_B", "s_AB_noise_max",
    }
    for i, rec in enumerate(sweep):
        missing = required_keys - set(rec.keys())
        if missing:
            raise KeyError(f"sweep[{i}] missing keys: {missing}")

    # --- helpers ---
    def fmt_cos(c: float) -> str:
        """Format 0.0 -> '0p0', 0.1 -> '0p1', 1.0 -> '1p0'."""
        return f"{c:.1f}".replace(".", "p")

    # --- per-slice metrics ---
    metrics: dict[str, float | int] = {"version": 1}

    or_sharpness_vals = []
    linear_sharpness_vals = []

    for rec in sweep:
        cos = rec["cos"]
        tag = fmt_cos(cos)

        # Ideal OR denominator: max of the two single-query signal scores
        denom_or = max(rec["s_A_at_A"], rec["s_B_at_B"])

        # OR sharpness: min(s_AB_at_A, s_AB_at_B) / denom_or
        num_or = min(rec["s_AB_at_A"], rec["s_AB_at_B"])
        sharp_or = _safe_div(num_or, denom_or)
        metrics[f"or_sharpness_cos_{tag}"] = sharp_or
        or_sharpness_vals.append(sharp_or)

        # Linear baseline: s_lin = s_A + s_B
        s_lin_at_A = rec["s_A_at_A"] + rec["s_B_at_A"]
        s_lin_at_B = rec["s_A_at_B"] + rec["s_B_at_B"]
        num_lin = min(s_lin_at_A, s_lin_at_B)
        sharp_lin = _safe_div(num_lin, denom_or)
        metrics[f"linear_baseline_sharpness_cos_{tag}"] = sharp_lin
        linear_sharpness_vals.append(sharp_lin)

        # Noise leakage for the combined query
        denom_noise = max(rec["s_AB_at_A"], rec["s_AB_at_B"])
        leakage = _safe_div(rec["s_AB_noise_max"], denom_noise)
        metrics[f"or_noise_leakage_cos_{tag}"] = leakage

    # --- headline summary (canonical condition = first sweep entry, cos=0.0) ---
    metrics["or_sharpness_canonical"] = or_sharpness_vals[0] if or_sharpness_vals else 0.0

    # --- lift over linear at canonical ---
    if linear_sharpness_vals:
        metrics["lift_over_linear_canonical"] = or_sharpness_vals[0] - linear_sharpness_vals[0]
    else:
        metrics["lift_over_linear_canonical"] = 0.0

    # --- superposition robustness: worst-case sharpness relative to canonical ---
    if or_sharpness_vals and or_sharpness_vals[0] > 0:
        metrics["superposition_robustness"] = min(or_sharpness_vals) / or_sharpness_vals[0]
    else:
        metrics["superposition_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: return True if the metrics indicate a degenerate/broken run.
    Used to short-circuit the jury stage.
    """
    # NaN / inf math failures.
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # The combined query attends to at most one signal key (or none) -> there is
    # no OR behaviour at all. Note: the linear-superposition reference
    # (linear_baseline_sharpness_*) is an *upper* oracle here, not a beatable
    # strawman, so it must NOT be used as a "broken" threshold -- doing so would
    # flag even a perfect max-pooling attempt. Only the degenerate floor is safe.
    sharp = metrics.get("or_sharpness_canonical")
    if isinstance(sharp, (int, float)) and sharp <= 0.0:
        return True

    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
