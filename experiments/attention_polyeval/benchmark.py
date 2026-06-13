import math
from typing import Any

VERSION = 1


def score(payload: dict) -> dict[str, float | int]:
    """Compute metrics from task payload."""
    _validate_payload(payload)
    
    sweep = payload["sweep"]
    linear_baseline = payload["linear_baseline"]
    config = payload["config"]
    
    # Build lookup by degree
    sweep_by_deg = {rec["degree"]: rec for rec in sweep}
    baseline_by_deg = {rec["degree"]: rec for rec in linear_baseline}
    
    degrees = config["degrees"]
    canonical_degree = 2  # degree=2 is the canonical non-linear test
    
    metrics: dict[str, float | int] = {"version": payload["version"]}
    
    # Per-degree metrics
    for deg in degrees:
        s = sweep_by_deg[deg]
        b = baseline_by_deg[deg]
        
        # Model metrics
        metrics[f"poly_mse_degree_{deg}"] = s["mse"]
        metrics[f"poly_correlation_degree_{deg}"] = s["correlation"]
        metrics[f"poly_r2_degree_{deg}"] = s["variance_explained"]
        
        # Linear baseline metrics
        metrics[f"linear_baseline_mse_degree_{deg}"] = b["mse"]
        metrics[f"linear_baseline_correlation_degree_{deg}"] = b["correlation"]
        metrics[f"linear_baseline_r2_degree_{deg}"] = b["variance_explained"]
        
        # Lift over baseline (positive = model beats baseline)
        metrics[f"nonlinear_lift_mse_degree_{deg}"] = b["mse"] - s["mse"]
        metrics[f"nonlinear_lift_r2_degree_{deg}"] = s["variance_explained"] - b["variance_explained"]
    
    # Canonical (headline) metrics at degree=2
    canon = sweep_by_deg[canonical_degree]
    canon_base = baseline_by_deg[canonical_degree]
    
    metrics["poly_mse_canonical"] = canon["mse"]
    metrics["poly_correlation_canonical"] = canon["correlation"]
    metrics["poly_r2_canonical"] = canon["variance_explained"]
    metrics["linear_baseline_mse_canonical"] = canon_base["mse"]
    metrics["linear_baseline_correlation_canonical"] = canon_base["correlation"]
    metrics["linear_baseline_r2_canonical"] = canon_base["variance_explained"]
    metrics["nonlinear_lift_mse_canonical"] = canon_base["mse"] - canon["mse"]
    metrics["nonlinear_lift_r2_canonical"] = canon["variance_explained"] - canon_base["variance_explained"]
    
    # Headline: improvement over linear baseline on canonical degree-2 task
    metrics["poly_eval_headline"] = metrics["nonlinear_lift_r2_canonical"]
    
    return metrics


def _validate_payload(payload: dict) -> None:
    """Validate payload structure, raise ValueError/KeyError on violation."""
    required_keys = ["version", "config", "sweep", "linear_baseline"]
    for key in required_keys:
        if key not in payload:
            raise KeyError(f"payload missing required key {key!r}")
    
    if not isinstance(payload["version"], int):
        raise ValueError(f"payload['version'] must be int, got {type(payload['version'])}")
    
    sweep = payload["sweep"]
    baseline = payload["linear_baseline"]
    
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("payload['sweep'] must be a non-empty list")
    if not isinstance(baseline, list) or not baseline:
        raise ValueError("payload['linear_baseline'] must be a non-empty list")
    if len(sweep) != len(baseline):
        raise ValueError(f"sweep length {len(sweep)} != baseline length {len(baseline)}")
    
    required_rec_keys = ["degree", "mse", "correlation", "variance_explained"]
    for i, (s, b) in enumerate(zip(sweep, baseline)):
        for rec, name in [(s, "sweep"), (b, "linear_baseline")]:
            for k in required_rec_keys:
                if k not in rec:
                    raise KeyError(f"payload[{name}][{i}] missing key {k!r}")
                if not isinstance(rec[k], (int, float)):
                    raise ValueError(f"payload[{name}][{i}][{k!r}] must be numeric")
            if s["degree"] != b["degree"]:
                raise ValueError(f"degree mismatch at index {i}: sweep={s['degree']} vs baseline={b['degree']}")


def is_obviously_broken(metrics: dict) -> bool:
    """Return True if metrics indicate a fundamentally broken attempt (skip jury)."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    
    # Headline should beat linear baseline (positive lift)
    headline = metrics.get("poly_eval_headline")
    if isinstance(headline, (int, float)) and headline <= 0:
        return True
    
    # Canonical R² should be reasonable (not worse than random)
    canon_r2 = metrics.get("poly_r2_canonical")
    if isinstance(canon_r2, (int, float)) and canon_r2 < -0.5:
        return True
    
    return False