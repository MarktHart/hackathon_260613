"""Benchmark for attention_and: scores AND sharpness and superposition robustness."""

from __future__ import annotations

import math
from typing import Any

VERSION = 2

# Number of GPU slots the experiment subprocess needs (default 1).
# This goal is pure Python/NumPy on tiny tensors; no GPU required.
GPU_REQUIREMENT = 0


def _cos_to_key(cos_val: float) -> str:
    """Convert a cosine float to the metric key suffix, e.g. 0.0 -> '0p0', -0.8 -> 'n0p8'."""
    if cos_val == 0.0:
        return "0p0"
    sign = "n" if cos_val < 0 else ""
    abs_val = abs(cos_val)
    # One decimal place, replace '.' with 'p'
    return f"{sign}{abs_val:.1f}".replace(".", "p")


def _linear_baseline_weight(cos_val: float) -> float:
    """Weight a linear (non-AND) superposition puts on the midpoint key.
    
    For a query q = α f_A + β f_B with α²+β²=1, cos = 2αβ.
    The midpoint key is (f_A+f_B)/√2. Linear attention weight ∝ ⟨q, k_AND⟩²
    = (α+β)²/2 = (1+2αβ)/2 = (1+cos)/2.
    But we want weight on k_AND *relative to k_A and k_B*. The linear baseline
    for the AND key is simply the interpolation: at cos=-1 (pure A or B) weight 0,
    at cos=1 (A=B) weight 1, at cos=0 weight 0.5.
    """
    return (1.0 + cos_val) / 2.0


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """Compute all metrics from the payload produced by task.evaluate.
    
    Args:
        payload: dict with keys: version, model_name, sweep (list of 11 dicts),
                 canonical_cos.
    
    Returns:
        Flat dict of metric_name -> scalar (float or int).
    
    Raises:
        ValueError: if payload contract is violated.
        KeyError: if required keys are missing.
    """
    # ---- Contract validation ----
    required_keys = ("version", "model_name", "sweep", "canonical_cos")
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"payload missing required key: {k}")

    if payload["version"] != VERSION:
        raise ValueError(f"payload version {payload['version']} != benchmark VERSION {VERSION}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) != 11:
        raise ValueError(f"sweep must be a list of 11 records, got {len(sweep) if isinstance(sweep, list) else type(sweep)}")

    # Expected cosines in order
    expected_cosines = [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    and_weights = []
    for i, (rec, exp_cos) in enumerate(zip(sweep, expected_cosines)):
        for f in ("cos_qA_qB", "and_weight", "a_weight", "b_weight"):
            if f not in rec:
                raise KeyError(f"sweep[{i}] missing field: {f}")
        cos_val = rec["cos_qA_qB"]
        if not math.isclose(cos_val, exp_cos, abs_tol=1e-6):
            raise ValueError(f"sweep[{i}] cosine {cos_val} != expected {exp_cos}")
        w = rec["and_weight"]
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"sweep[{i}] and_weight {w} not in [0, 1]")
        and_weights.append(w)

    canonical_cos = payload["canonical_cos"]
    if not math.isclose(canonical_cos, 0.0, abs_tol=1e-6):
        raise ValueError(f"canonical_cos {canonical_cos} != 0.0")

    # ---- Compute metrics ----
    metrics: dict[str, float | int] = {}
    metrics["version"] = VERSION

    # Per-slice sharpness and linear baseline
    for cos_val, w in zip(expected_cosines, and_weights):
        suffix = _cos_to_key(cos_val)
        metrics[f"and_sharpness_cos_{suffix}"] = w
        baseline = _linear_baseline_weight(cos_val)
        metrics[f"linear_baseline_sharpness_cos_{suffix}"] = baseline
        metrics[f"lift_over_linear_baseline_cos_{suffix}"] = w - baseline

    # Canonical sharpness (at cos = 0.0)
    canonical_idx = expected_cosines.index(0.0)
    metrics["and_sharpness_canonical"] = and_weights[canonical_idx]

    # Superposition robustness: min / max across the sweep
    # Higher = more robust (sharp peak at orthogonality, low elsewhere)
    min_w = min(and_weights)
    max_w = max(and_weights)
    if max_w == 0.0:
        metrics["superposition_robustness"] = 1.0  # all zero → flat (degenerate)
    else:
        metrics["superposition_robustness"] = min_w / max_w

    return metrics


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """Return True if the metrics indicate a clearly failed attempt.
    
    Used by the pipeline to skip the expensive jury stage.
    """
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Superposition robustness should be < 1.0 for a non-degenerate mechanism.
    # A linear baseline has robustness ≈ 1.0 (flat). A good AND has low min/max.
    # Flag if robustness >= 0.9 (too flat) OR canonical sharpness <= linear baseline at cos=0.
    robustness = metrics.get("superposition_robustness")
    canonical_sharp = metrics.get("and_sharpness_canonical")
    linear_baseline_canonical = metrics.get("linear_baseline_sharpness_cos_0p0")

    if isinstance(robustness, (int, float)) and robustness >= 0.9:
        return True
    if isinstance(canonical_sharp, (int, float)) and isinstance(linear_baseline_canonical, (int, float)):
        if canonical_sharp <= linear_baseline_canonical:
            return True

    return False