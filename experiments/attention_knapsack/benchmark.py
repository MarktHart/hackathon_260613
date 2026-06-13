"""Benchmark scoring for the attention_knapsack goal.

Pure Python. No numpy, no imports from any attempt directory. Consumes the
payload produced by task.evaluate() and returns a flat dict of scalar metrics.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs on the GPU; 1 slot is the minimum.
GPU_REQUIREMENT = 1

_EXPECTED_FRACS = [0.3, 0.4, 0.5, 0.6, 0.7]
_CANONICAL_FRAC = 0.5


def _frac_key(prefix: str, frac: float) -> str:
    """Slice key, e.g. ('knapsack_optimality', 0.7) -> 'knapsack_optimality_cap_0p7'."""
    return f"{prefix}_cap_{frac:.1f}".replace(".", "p")


def _optimality(gap) -> float:
    """Convert an optimality_gap into a clamped optimality in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - float(gap)))


def _require(rec, key, what):
    if key not in rec:
        raise KeyError(f"{what} missing required key: {key!r}")
    return rec[key]


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload shape::

        {
          "version": 1,
          "config": {...},
          "canonical":          {capacity_frac, optimal_value, model_value,
                                 model_weight, feasible_rate, optimality_gap},
          "sweep":          [ <record>, ... 5 records over capacity_frac ],
          "baseline_canonical": <record>,
          "baseline_sweep": [ <record>, ... 5 records ],
        }

    Direction of better: all metrics are bigger-is-better except the explicit
    *_gap fields are not emitted — optimality (1 - gap) is reported instead.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"Unsupported payload version: {payload.get('version')}. Expected {VERSION}."
        )

    for k in ("canonical", "sweep", "baseline_canonical", "baseline_sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    canon = payload["canonical"]
    sweep = payload["sweep"]
    base_canon = payload["baseline_canonical"]
    base_sweep = payload["baseline_sweep"]

    if not isinstance(sweep, list) or len(sweep) != len(_EXPECTED_FRACS):
        raise ValueError(f"sweep must be a list of {len(_EXPECTED_FRACS)} records")
    if not isinstance(base_sweep, list) or len(base_sweep) != len(_EXPECTED_FRACS):
        raise ValueError(
            f"baseline_sweep must be a list of {len(_EXPECTED_FRACS)} records"
        )

    metrics: dict[str, float | int] = {"version": VERSION}

    # ─── Canonical ───
    opt_canon = _optimality(_require(canon, "optimality_gap", "canonical"))
    feas_canon = float(_require(canon, "feasible_rate", "canonical"))
    base_opt_canon = _optimality(
        _require(base_canon, "optimality_gap", "baseline_canonical")
    )
    metrics["knapsack_optimality_canonical"] = opt_canon
    metrics["knapsack_feasible_canonical"] = feas_canon
    metrics["linear_baseline_optimality_canonical"] = base_opt_canon
    metrics["lift_over_linear_baseline"] = opt_canon - base_opt_canon

    # ─── Sweep per-slice ───
    opt_sweep_vals = []
    for i, (rec, brec) in enumerate(zip(sweep, base_sweep)):
        if not isinstance(rec, dict) or not isinstance(brec, dict):
            raise ValueError(f"sweep records at index {i} must be dicts")
        frac = _EXPECTED_FRACS[i]
        # Cross-check the record's own capacity_frac if present.
        rec_frac = rec.get("capacity_frac")
        if rec_frac is not None and abs(float(rec_frac) - frac) > 1e-3:
            raise ValueError(
                f"sweep[{i}] capacity_frac mismatch: expected {frac}, got {rec_frac}"
            )
        opt = _optimality(_require(rec, "optimality_gap", f"sweep[{i}]"))
        feas = float(_require(rec, "feasible_rate", f"sweep[{i}]"))
        base_opt = _optimality(_require(brec, "optimality_gap", f"baseline_sweep[{i}]"))
        opt_sweep_vals.append(opt)
        metrics[_frac_key("knapsack_optimality", frac)] = opt
        metrics[_frac_key("knapsack_feasible", frac)] = feas
        metrics[_frac_key("linear_baseline_optimality", frac)] = base_opt

    # ─── Headline: mean optimality across the capacity sweep ───
    if opt_sweep_vals:
        metrics["knapsack_optimality_robustness"] = sum(opt_sweep_vals) / len(
            opt_sweep_vals
        )
    else:
        metrics["knapsack_optimality_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical degeneracy check to short-circuit the jury. Conservative:
    only fires on clear failures (NaN/inf, sub-baseline, near-zero feasibility).
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    opt = metrics.get("knapsack_optimality_canonical")
    base = metrics.get("linear_baseline_optimality_canonical")
    if isinstance(opt, (int, float)) and isinstance(base, (int, float)):
        # Not beating the no-mechanism greedy baseline => nothing learned.
        if opt <= base:
            return True

    feas = metrics.get("knapsack_feasible_canonical")
    if isinstance(feas, (int, float)) and feas < 0.01:
        return True

    return False
