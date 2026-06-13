"""Benchmark for the `attention_equality` goal.

Consumes the payload returned by `task.evaluate` and produces a flat dict of
named scalar metrics.  Pure, deterministic, side-effect free.  See README.md
for definitions.

Direction-of-better: every `match_mass` / `equality_*` / `lift_*` metric is
**bigger-is-better** (perfect equality lookup -> 1.0).  `attn_rowsum_max_dev`
is a sanity diagnostic, smaller-is-better (0 == perfectly row-stochastic).
"""

from __future__ import annotations

import math
from typing import Dict, Union

VERSION = 1

# Synthetic NumPy task — no model on the GPU.
GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU

Number = Union[float, int]

_REQUIRED_SLICE_KEYS = ("L", "match_mass", "uniform_baseline", "n_eval")


def _fnum(x) -> float:
    return float(x)


def score(payload: dict) -> Dict[str, Number]:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for key in ("version", "sweep", "canonical"):
        if key not in payload:
            raise KeyError(f"payload missing required key: {key!r}")

    if payload["version"] != VERSION:
        raise ValueError(
            f"Unsupported payload version {payload['version']!r}; expected {VERSION}"
        )

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or not sweep:
        raise ValueError("payload['sweep'] must be a non-empty list")

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        for key in _REQUIRED_SLICE_KEYS:
            if key not in rec:
                raise KeyError(f"sweep record missing required key: {key!r}")

    canonical = payload["canonical"]
    if not isinstance(canonical, dict) or "match_mass" not in canonical:
        raise ValueError("payload['canonical'] must be a dict with 'match_mass'")

    metrics: Dict[str, Number] = {"version": VERSION}

    # --- Per-slice values (keyed by integer L). ---
    match_vals = []
    baseline_vals = []
    for rec in sweep:
        L = int(rec["L"])
        mm = _fnum(rec["match_mass"])
        ub = _fnum(rec["uniform_baseline"])
        metrics[f"match_mass_L_{L}"] = mm
        metrics[f"uniform_baseline_L_{L}"] = ub
        metrics[f"lift_over_uniform_L_{L}"] = mm - ub
        match_vals.append(mm)
        baseline_vals.append(ub)

    # --- Canonical condition. ---
    canon_mm = _fnum(canonical["match_mass"])
    canon_ub = _fnum(canonical.get("uniform_baseline", 0.0))
    metrics["match_mass_canonical"] = canon_mm
    metrics["uniform_baseline_canonical"] = canon_ub
    metrics["lift_over_uniform_canonical"] = canon_mm - canon_ub

    # --- Headline: mean match_mass across the sweep, in [0, 1]. ---
    n = len(match_vals)
    metrics["equality_robustness"] = float(sum(match_vals) / n) if n else 0.0
    metrics["uniform_baseline_robustness"] = (
        float(sum(baseline_vals) / n) if n else 0.0
    )

    # --- Sanity diagnostic. ---
    metrics["attn_rowsum_max_dev"] = _fnum(payload.get("attn_rowsum_max_dev", 0.0))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Short-circuit the jury for mechanically-degenerate attempts."""
    # NaN / inf anywhere => degenerate math failure.
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Attention that doesn't sum to ~1 over allowed keys is malformed.
    dev = metrics.get("attn_rowsum_max_dev")
    if isinstance(dev, (int, float)) and dev > 0.1:
        return True

    # Must beat the uniform baseline at the canonical condition; a head that
    # doesn't route extra mass onto the matching key isn't computing equality.
    mm = metrics.get("match_mass_canonical")
    ub = metrics.get("uniform_baseline_canonical")
    if isinstance(mm, (int, float)) and isinstance(ub, (int, float)):
        if mm <= ub * 1.5:
            return True

    return False
