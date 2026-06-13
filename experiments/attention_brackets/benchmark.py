"""Benchmark for the attention_brackets goal.

Pure Python. Deterministic. Side-effect free. No imports from any attempt dir.

The payload (from task.evaluate) is a `sweep` over the maximum nesting depth of
the generated bracket sequences. For each depth we know:

  - match_accuracy        : fraction of closing brackets whose attention argmax
                            lands on the true matching opener (the parser pop);
  - match_mass            : average attention mass placed on that opener;
  - uniform_baseline_mass : average mass a uniform causal head would place there.

We score how strongly the head implements stack-matching (vs. uniform routing)
and how that holds up as nesting deepens.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1

# Must match task.py.
DEPTHS: tuple[int, ...] = (1, 2, 3, 4, 5)
CANONICAL_DEPTH: int = 3

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU

_REC_KEYS = ("depth", "n_closers", "match_accuracy", "match_mass", "uniform_baseline_mass")


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return _num(v) and not (math.isnan(v) or math.isinf(v))


def _lift(mass: float, baseline: float) -> float:
    """Normalised lift over uniform routing, in [0, 1].

        lift = (mass - baseline) / (1 - baseline)

    1.0 => all attention on the matching opener; 0.0 => no better than uniform;
    negative is clamped to 0. If baseline is ~1 (degenerate, very short rows)
    there is no headroom, so return 0."""
    headroom = 1.0 - baseline
    if headroom <= 1e-9:
        return 0.0
    return max(0.0, min(1.0, (mass - baseline) / headroom))


def _validate(payload: Any) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}")
    sweep = payload.get("sweep")
    if not isinstance(sweep, list) or len(sweep) != len(DEPTHS):
        raise ValueError(f"payload['sweep'] must be a list of length {len(DEPTHS)}")
    for i, (expected_d, rec) in enumerate(zip(DEPTHS, sweep)):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        for k in _REC_KEYS:
            if k not in rec:
                raise KeyError(f"sweep[{i}] missing key {k!r}")
        if rec["depth"] != expected_d:
            raise ValueError(f"sweep[{i}]['depth'] = {rec['depth']!r}, expected {expected_d}")
        for k in ("match_accuracy", "match_mass", "uniform_baseline_mass"):
            if not _finite(rec[k]):
                raise ValueError(f"sweep[{i}][{k!r}] must be finite, got {rec[k]!r}")
        if not isinstance(rec["n_closers"], int) or rec["n_closers"] < 0:
            raise ValueError(f"sweep[{i}]['n_closers'] must be a non-negative int")
    return sweep


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    sweep = _validate(payload)

    by_depth = {rec["depth"]: rec for rec in sweep}
    metrics: dict[str, float | int] = {"version": VERSION}

    lifts: list[float] = []
    for d in DEPTHS:
        rec = by_depth[d]
        lift = _lift(rec["match_mass"], rec["uniform_baseline_mass"])
        lifts.append(lift)
        metrics[f"match_accuracy_depth_{d}"] = float(rec["match_accuracy"])
        metrics[f"match_mass_depth_{d}"] = float(rec["match_mass"])
        metrics[f"uniform_baseline_mass_depth_{d}"] = float(rec["uniform_baseline_mass"])
        metrics[f"match_lift_depth_{d}"] = float(lift)

    # ---- canonical convenience metrics ----
    c = by_depth[CANONICAL_DEPTH]
    metrics["bracket_match_accuracy_canonical"] = float(c["match_accuracy"])
    metrics["bracket_match_mass_canonical"] = float(c["match_mass"])
    metrics["uniform_baseline_mass_canonical"] = float(c["uniform_baseline_mass"])
    metrics["lift_over_uniform_canonical"] = _lift(c["match_mass"], c["uniform_baseline_mass"])

    # ---- headline: how well stack-matching survives the depth sweep ----
    # The worst slice relative to a perfect head: min normalised lift across all
    # depths. 1.0 => perfect matching at every nesting depth. In [0, 1].
    metrics["bracket_match_robustness"] = float(min(lifts)) if lifts else 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical short-circuit before the (expensive) jury. Never True for a
    borderline-but-real result — only clearly degenerate ones."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # A real matching head must clear the uniform floor at the canonical depth.
    mass = metrics.get("bracket_match_mass_canonical")
    baseline = metrics.get("uniform_baseline_mass_canonical")
    if _num(mass) and _num(baseline):
        if mass <= baseline * 1.25:
            return True
    return False
