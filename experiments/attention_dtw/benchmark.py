"""Benchmark for attention_dtw.

Pure Python / no I/O. Consumes the payload from task.evaluate and returns a
flat dict of scalar metrics. Bigger is better for every metric.
"""

from __future__ import annotations

import math

VERSION = 1

# Pipeline-only hook: GPU slots the attempt subprocess needs (minimum 1).
GPU_REQUIREMENT = 1


def _wkey(prefix: str, warp: float) -> str:
    """Slice key, e.g. ('path_overlap', 0.5) -> 'path_overlap_warp_0p5'."""
    return f"{prefix}_warp_{warp:g}".replace(".", "p").replace("-", "neg")


def _index(records, what):
    out = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"Each {what} record must be a dict")
        if "warp" not in rec:
            raise KeyError(f"{what} record missing 'warp'")
        out[round(float(rec["warp"]), 6)] = rec
    return out


def score(payload: dict) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "canonical_warp", "warp_sweep", "sweep", "baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    warp_sweep = payload["warp_sweep"]
    if not isinstance(warp_sweep, (list, tuple)) or len(warp_sweep) == 0:
        raise ValueError("payload['warp_sweep'] must be a non-empty list")
    warp_sweep = [float(w) for w in warp_sweep]

    sweep = payload["sweep"]
    baseline = payload["baseline"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(warp_sweep):
        raise ValueError("payload['sweep'] must be a list matching warp_sweep length")
    if not isinstance(baseline, (list, tuple)) or len(baseline) != len(warp_sweep):
        raise ValueError("payload['baseline'] must be a list matching warp_sweep length")

    sweep_by = _index(sweep, "sweep")
    base_by = _index(baseline, "baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    for warp in warp_sweep:
        key = round(warp, 6)
        s = sweep_by.get(key, {})
        b = base_by.get(key, {})

        metrics[_wkey("path_overlap", warp)] = float(s.get("best_head_overlap", 0.0))
        metrics[_wkey("mean_head_overlap", warp)] = float(s.get("mean_head_overlap", 0.0))
        metrics[_wkey("monotonicity", warp)] = float(s.get("monotonicity", 0.0))
        metrics[_wkey("diagonal_baseline_overlap", warp)] = float(b.get("diagonal_overlap", 0.0))
        metrics[_wkey("uniform_baseline_overlap", warp)] = float(b.get("uniform_overlap", 0.0))

    # --- Canonical condition ---
    canonical = float(payload["canonical_warp"])
    canon_key = _wkey("path_overlap", canonical)
    metrics["path_overlap_canonical"] = float(metrics.get(canon_key, 0.0))

    diag_canon = float(metrics.get(_wkey("diagonal_baseline_overlap", canonical), 0.0))
    metrics["lift_over_diagonal_canonical"] = (
        metrics["path_overlap_canonical"] - diag_canon
    )

    # --- Headline: alignment_robustness ---
    # Overlap retained at the largest warp relative to the smallest (no-warp).
    low = float(metrics.get(_wkey("path_overlap", warp_sweep[0]), 0.0))
    high = float(metrics.get(_wkey("path_overlap", warp_sweep[-1]), 0.0))
    robustness = high / low if low > 1e-12 else 0.0
    metrics["alignment_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True on clearly degenerate metrics, so the pipeline can skip the jury.

    Never True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    sharp = metrics.get("path_overlap_canonical")
    chance = metrics.get("uniform_baseline_overlap_warp_0p5")  # canonical warp = 0.5
    if isinstance(sharp, (int, float)) and isinstance(chance, (int, float)):
        if sharp <= chance:
            return True

    return False
