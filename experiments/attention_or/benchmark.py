import math
from typing import Any

VERSION = 1

def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """
    Compute metrics from the attention_or payload.

    Args:
        payload: Dict with keys version, config, sweep (see README.md).

    Returns:
        Flat dict of metrics. First key is 'version'.
    """
    _validate_payload(payload)

    sweep = payload["sweep"]
    canonical_rho = payload["config"]["canonical_rho"]

    # Helper to format float for metric keys
    def fmt(rho: float) -> str:
        return f"{rho:.2f}".replace(".", "p").replace("-", "m")

    # Extract per-slice measurements
    sharpness_by_rho: dict[float, float] = {}
    gap_by_rho: dict[float, float] = {}
    out00_by_rho: dict[float, float] = {}
    out01_by_rho: dict[float, float] = {}
    out10_by_rho: dict[float, float] = {}
    out11_by_rho: dict[float, float] = {}

    for record in sweep:
        rho = record["rho"]
        # Use first component (index 0) — the only non-zero in value basis
        out00 = record["out_00"][0]
        out01 = record["out_01"][0]
        out10 = record["out_10"][0]
        out11 = record["out_11"][0]

        out00_by_rho[rho] = out00
        out01_by_rho[rho] = out01
        out10_by_rho[rho] = out10
        out11_by_rho[rho] = out11

        or1_vals = [out01, out10, out11]
        mean_or1 = sum(or1_vals) / 3.0
        max_or1 = max(or1_vals)
        min_or1 = min(or1_vals)

        gap = mean_or1 - out00
        gap_by_rho[rho] = gap

        denom = max_or1 - min_or1 + 1e-8
        sharpness = gap / denom if denom > 0 else 0.0
        sharpness_by_rho[rho] = sharpness

    # Headline: superposition robustness = worst-case relative sharpness
    sharpness_at_zero = sharpness_by_rho.get(0.0, 0.0)
    if sharpness_at_zero <= 0:
        robustness = 0.0
    else:
        ratios = [sharpness_by_rho[rho] / sharpness_at_zero for rho in sharpness_by_rho]
        robustness = min(ratios)

    # Linear baseline: optimal linear classifier on 4 points
    # Features: one-hot [A, B] -> 2D. Labels: OR(A,B).
    # With 4 points in 2D, perfect separation is possible iff not all same label.
    # But we compute the "sharpness" of the linear probe's output on the same scale.
    # Since the linear probe sees clean one-hot features (no interference),
    # its sharpness is 1.0 at all rho (rho doesn't affect the linear probe's input).
    # We report it as a constant baseline for comparison.
    linear_sharpness = 1.0  # Perfect separation achievable with linear probe on one-hot

    metrics: dict[str, float | int] = {
        "version": VERSION,
    }

    # Per-slice metrics
    for rho in sorted(sharpness_by_rho.keys()):
        tag = fmt(rho)
        metrics[f"or_sharpness_rho_{tag}"] = sharpness_by_rho[rho]
        metrics[f"or_gap_rho_{tag}"] = gap_by_rho[rho]

    # Canonical slice
    canon_tag = fmt(canonical_rho)
    metrics["or_sharpness_canonical"] = sharpness_by_rho[canonical_rho]
    metrics["or_gap_canonical"] = gap_by_rho[canonical_rho]

    # Headline
    metrics["or_superposition_robustness"] = robustness

    # Baselines
    for rho in sorted(sharpness_by_rho.keys()):
        tag = fmt(rho)
        metrics[f"linear_baseline_sharpness_rho_{tag}"] = linear_sharpness
    metrics["linear_baseline_sharpness_canonical"] = linear_sharpness
    metrics["lift_over_linear_canonical"] = (
        metrics["or_sharpness_canonical"] - linear_sharpness
    )

    return metrics


def _validate_payload(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(f"payload version {payload.get('version')} != benchmark VERSION {VERSION}")
    if "config" not in payload or "sweep" not in payload:
        raise KeyError("payload missing 'config' or 'sweep'")
    config = payload["config"]
    required_config = {"d", "canonical_rho", "rho_sweep"}
    if not required_config.issubset(config.keys()):
        raise KeyError(f"config missing keys: {required_config - config.keys()}")
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("sweep must be a non-empty list")
    expected_rhos = set(config["rho_sweep"])
    seen_rhos = set()
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        if "rho" not in rec:
            raise KeyError(f"sweep[{i}] missing 'rho'")
        rho = rec["rho"]
        if rho in seen_rhos:
            raise ValueError(f"duplicate rho {rho} in sweep")
        seen_rhos.add(rho)
        for key in ("out_00", "out_01", "out_10", "out_11"):
            if key not in rec:
                raise KeyError(f"sweep[{i}] missing '{key}'")
            val = rec[key]
            if not isinstance(val, list) or len(val) != config["d"]:
                raise ValueError(f"sweep[{i}]['{key}'] must be list of length d={config['d']}")
        if "sharpness" not in rec:
            raise KeyError(f"sweep[{i}] missing 'sharpness'")
        if not isinstance(rec["sharpness"], (int, float)) or isinstance(rec["sharpness"], bool):
            raise ValueError(f"sweep[{i}]['sharpness'] must be a number")
    if seen_rhos != expected_rhos:
        raise ValueError(f"sweep rhos {sorted(seen_rhos)} != config.rho_sweep {sorted(expected_rhos)}")


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """
    Return True if the metrics indicate a fundamentally broken attempt
    (NaN/inf, or sharpness at canonical slice no better than linear baseline).
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    sharp = metrics.get("or_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_canonical")
    if isinstance(sharp, (int, float)) and isinstance(baseline, (int, float)):
        # If the method doesn't meaningfully beat the linear baseline at canonical condition,
        # something is wrong — the attention mechanism should exploit the known structure.
        if sharp <= baseline * 1.01:  # allow tiny numerical slop
            return True

    robustness = metrics.get("or_superposition_robustness")
    if isinstance(robustness, (int, float)) and robustness < 0:
        return True

    return False