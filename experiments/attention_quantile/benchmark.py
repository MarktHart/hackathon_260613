VERSION = 1

import math
from typing import Any

def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """Compute metrics from the attention quantile payload."""
    _validate_payload(payload)
    
    sweep = payload["sweep"]
    config = payload["config"]
    n_keys = config["n_keys"]
    
    # Baseline: uniform attention gives ratio = 1.0
    linear_baseline_ratio = 1.0
    
    metrics: dict[str, float | int] = {"version": VERSION}
    
    # Per-slice metrics
    pareto_ratios = []
    exp_ratios = []
    
    for rec in sweep:
        cid = rec["condition_id"]
        ratio = rec["quantile_ratio"]
        
        if rec["tail_type"] == "pareto":
            metrics[f"quantile_ratio_{cid}"] = ratio
            pareto_ratios.append(ratio)
        else:
            metrics[f"quantile_ratio_{cid}"] = ratio
            exp_ratios.append(ratio)
    
    # Headline: canonical is pareto_0p5 (α=0.5)
    canonical_ratio = None
    for rec in sweep:
        if rec["condition_id"] == "pareto_0p5":
            canonical_ratio = rec["quantile_ratio"]
            break
    
    if canonical_ratio is None:
        raise ValueError("Canonical condition pareto_0p5 not found in sweep")
    
    metrics["quantile_ratio_canonical"] = canonical_ratio
    
    # Aggregate lift metrics. Guard the denominator: an empty exponential slice
    # or a mean of exactly 0 (degenerate ultra-sparse attention whose 90th
    # percentile is 0 in every light-tail condition) would otherwise divide by
    # zero. Emit nan in that case, which is_obviously_broken() catches.
    if pareto_ratios and exp_ratios and np.mean(exp_ratios) > 0:
        metrics["pareto_vs_exponential_lift"] = float(np.mean(pareto_ratios) / np.mean(exp_ratios))
    else:
        metrics["pareto_vs_exponential_lift"] = float('nan')
    
    metrics["linear_baseline_quantile_ratio_canonical"] = linear_baseline_ratio
    metrics["lift_over_linear_baseline"] = canonical_ratio / linear_baseline_ratio if linear_baseline_ratio > 0 else float('inf')
    
    return metrics

def _validate_payload(payload: dict[str, Any]) -> None:
    if "version" not in payload:
        raise ValueError("payload missing 'version'")
    if payload["version"] != 1:
        raise ValueError(f"payload version {payload['version']} != expected 1")
    if "config" not in payload:
        raise ValueError("payload missing 'config'")
    if "sweep" not in payload:
        raise ValueError("payload missing 'sweep'")
    
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("sweep must be a non-empty list")
    
    expected_conditions = [
        "pareto_0p1", "pareto_0p3", "pareto_0p5", "pareto_0p7", "pareto_1p0",
        "exponential_0p5", "exponential_1p0", "exponential_2p0", "exponential_5p0",
    ]
    
    seen = set()
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] is not a dict")
        
        cid = rec.get("condition_id")
        if cid not in expected_conditions:
            raise ValueError(f"sweep[{i}]['condition_id'] = {cid!r}, expected one of {expected_conditions}")
        if cid in seen:
            raise ValueError(f"Duplicate condition_id: {cid}")
        seen.add(cid)
        
        tail_type = rec.get("tail_type")
        if tail_type not in ("pareto", "exponential"):
            raise ValueError(f"sweep[{i}]['tail_type'] = {tail_type!r}, expected 'pareto' or 'exponential'")
        
        alpha = rec.get("alpha")
        rate = rec.get("rate")
        
        if tail_type == "pareto":
            if alpha is None or not isinstance(alpha, (int, float)):
                raise ValueError(f"sweep[{i}]['alpha'] = {alpha!r}, expected float for pareto")
            if rate is not None:
                raise ValueError(f"sweep[{i}]['rate'] = {rate!r}, expected null for pareto")
        else:  # exponential
            if rate is None or not isinstance(rate, (int, float)):
                raise ValueError(f"sweep[{i}]['rate'] = {rate!r}, expected float for exponential")
            if alpha is not None:
                raise ValueError(f"sweep[{i}]['alpha'] = {alpha!r}, expected null for exponential")
        
        for key in ("quantile_50", "quantile_90", "quantile_ratio"):
            val = rec.get(key)
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"sweep[{i}]['{key}'] = {val!r}, expected finite float")
    
    if seen != set(expected_conditions):
        missing = set(expected_conditions) - seen
        raise ValueError(f"Missing conditions in sweep: {missing}")

def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """Pipeline hook: return True if metrics indicate a catastrophically failed run."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    
    # If canonical ratio is not > baseline (1.0), the method produces no structure
    canonical = metrics.get("quantile_ratio_canonical")
    baseline = metrics.get("linear_baseline_quantile_ratio_canonical")
    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        if canonical <= baseline * 1.01:  # allow tiny numerical noise
            return True
    
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU

# Need numpy for mean in score()
import numpy as np