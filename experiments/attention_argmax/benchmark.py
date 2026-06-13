"""Scoring for attention_argmax.

Pure Python, no external deps beyond stdlib. Deterministic, side-effect free.
"""
from __future__ import annotations

import math
from typing import Any


VERSION = 1


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """Compute all metrics from a task.py payload.

    Raises:
        KeyError: if required payload keys are missing.
        ValueError: if payload contains non-finite values or invalid structure.
    """
    # --- Validation ---------------------------------------------------------
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    required_keys = ("version", "config", "sweep", "baselines")
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"payload missing required key: {k}")

    if payload["version"] != VERSION:
        raise ValueError(f"payload version {payload['version']} != benchmark VERSION {VERSION}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    config = payload["config"]
    baselines = payload["baselines"]

    # Check sweep records have required fields
    expected_fields = (
        "separation", "winner_mass_mean", "winner_mass_std",
        "winner_rank_mean", "winner_rank_std",
        "entropy_mean", "entropy_std"
    )
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        for f in expected_fields:
            if f not in rec:
                raise KeyError(f"sweep[{i}] missing field: {f}")
            v = rec[f]
            if not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"sweep[{i}]['{f}'] must be a finite number, got {v}")

    # --- Helper to find canonical slice ------------------------------------
    canonical_sep = config.get("canonical_separation", 2.0)
    canonical_rec = None
    for rec in sweep:
        if abs(rec["separation"] - canonical_sep) < 1e-9:
            canonical_rec = rec
            break
    if canonical_rec is None:
        raise ValueError(f"No sweep record matches canonical_separation={canonical_sep}")

    # --- Extract per-slice values ------------------------------------------
    # Build lookup by separation (formatted for metric keys)
    def fmt_sep(x: float) -> str:
        return f"{x:.1f}".replace(".", "p")

    fidelity_by_sep = {}
    rank_by_sep = {}
    entropy_by_sep = {}

    for rec in sweep:
        key = fmt_sep(rec["separation"])
        fidelity_by_sep[key] = rec["winner_mass_mean"]
        rank_by_sep[key] = rec["winner_rank_mean"]
        entropy_by_sep[key] = rec["entropy_mean"]

    # --- Compute metrics ----------------------------------------------------
    metrics: dict[str, float | int] = {}
    metrics["version"] = VERSION

    # Headline: fidelity at canonical separation
    canonical_key = fmt_sep(canonical_sep)
    metrics["argmax_fidelity_canonical"] = fidelity_by_sep[canonical_key]

    # Per-slice fidelity
    for key, val in fidelity_by_sep.items():
        metrics[f"argmax_fidelity_sep_{key}"] = val

    # Per-slice rank (smaller is better)
    for key, val in rank_by_sep.items():
        metrics[f"argmax_rank_sep_{key}"] = val

    # Per-slice entropy (smaller is better)
    for key, val in entropy_by_sep.items():
        metrics[f"entropy_sep_{key}"] = val

    # Canonical rank and entropy
    metrics["argmax_rank_canonical"] = rank_by_sep[canonical_key]
    metrics["entropy_canonical"] = entropy_by_sep[canonical_key]

    # Robustness: how well fidelity at the HARD separation (0.5) holds up
    # relative to the EASY separation (4.0). A perfectly robust head keeps the
    # same fidelity at both → ratio 1; a head that degrades under hard
    # discrimination has hard < easy → ratio in [0, 1). Bigger is better.
    f_easy = fidelity_by_sep.get("4p0")
    f_hard = fidelity_by_sep.get("0p5")
    if f_easy is not None and f_hard is not None and f_easy > 0:
        metrics["selection_robustness"] = f_hard / f_easy
    else:
        metrics["selection_robustness"] = float("nan")

    # Baselines
    uniform_fidelity = baselines.get("uniform_winner_mass", 1.0 / config.get("N", 32))
    metrics["uniform_baseline_fidelity"] = uniform_fidelity
    metrics["lift_over_uniform_canonical"] = metrics["argmax_fidelity_canonical"] - uniform_fidelity

    return metrics


# --- Optional pipeline hooks ------------------------------------------------

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """Return True if metrics indicate a catastrophic failure (skip jury)."""
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Fidelity worse than uniform baseline (at canonical)
    fidelity = metrics.get("argmax_fidelity_canonical")
    baseline = metrics.get("uniform_baseline_fidelity")
    if isinstance(fidelity, (int, float)) and isinstance(baseline, (int, float)):
        # Allow small numerical tolerance
        if fidelity < baseline * 0.99:
            return True

    # Rank worse than random expectation (N+1)/2 ≈ 16.5 for N=32
    rank = metrics.get("argmax_rank_canonical")
    if isinstance(rank, (int, float)) and rank > 17:
        return True

    # Entropy higher than uniform (log N)
    entropy = metrics.get("entropy_canonical")
    uniform_entropy = math.log(32)
    if isinstance(entropy, (int, float)) and entropy > uniform_entropy * 1.01:
        return True

    return False