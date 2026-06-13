"""Benchmark for the attention_previous_token goal.

Consumes the payload from ``task.evaluate`` and returns a flat dict of scalar
metrics. Pure, deterministic, side-effect free. No imports from any attempt
directory. See README.md for the payload contract and metric definitions.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x: Any) -> bool:
    return _is_num(x) and not math.isnan(x) and not math.isinf(x)


def _fmt(v: float) -> str:
    """0.25 -> '0p25', 1.0 -> '1p00' (per the naming convention)."""
    return f"{float(v):.2f}".replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    # ---- Input validation ----
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )

    noise_sweep = payload.get("noise_sweep")
    if not isinstance(noise_sweep, (list, tuple)) or len(noise_sweep) == 0:
        raise ValueError("payload['noise_sweep'] must be a non-empty list")

    sweep = payload.get("sweep")
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(noise_sweep):
        raise ValueError(
            "payload['sweep'] must be a list the same length as 'noise_sweep'"
        )

    uniform_baseline = payload.get("uniform_baseline")
    if not _finite(uniform_baseline) or uniform_baseline <= 0:
        raise ValueError(
            f"payload['uniform_baseline'] must be a positive finite number, "
            f"got {uniform_baseline!r}"
        )

    canonical_noise = payload.get("canonical_noise")
    if not _finite(canonical_noise):
        raise ValueError(f"payload['canonical_noise'] must be finite, got {canonical_noise!r}")

    # ---- Index records by noise ----
    by_noise: dict[float, dict] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        noise = rec.get("noise")
        if not _finite(noise):
            raise KeyError("sweep record missing finite 'noise'")
        for k in ("prev_token_attention", "self_attention", "two_back_attention"):
            if not _finite(rec.get(k)):
                raise ValueError(f"sweep record (noise={noise}) has non-finite {k!r}")
        by_noise[float(noise)] = rec

    metrics: dict[str, float | int] = {"version": VERSION}
    metrics["uniform_baseline"] = float(uniform_baseline)

    # ---- Per-slice metrics ----
    for noise in noise_sweep:
        rec = by_noise.get(float(noise))
        val = float(rec["prev_token_attention"]) if rec is not None else 0.0
        metrics[f"prev_token_attn_noise_{_fmt(noise)}"] = val

    # ---- Canonical (headline) ----
    canon = by_noise.get(float(canonical_noise))
    prev_canonical = float(canon["prev_token_attention"]) if canon is not None else 0.0
    metrics["prev_token_attn_canonical"] = prev_canonical
    metrics["self_attn_canonical"] = float(canon["self_attention"]) if canon is not None else 0.0
    metrics["two_back_attn_canonical"] = float(canon["two_back_attention"]) if canon is not None else 0.0

    metrics["lift_over_uniform_canonical"] = prev_canonical - float(uniform_baseline)
    metrics["prev_token_lift_ratio_canonical"] = prev_canonical / float(uniform_baseline)

    # ---- Robustness: prev mass at max noise / at zero noise, clipped [0, 1] ----
    max_noise = max(float(n) for n in noise_sweep)
    rec_max = by_noise.get(max_noise)
    prev_max = float(rec_max["prev_token_attention"]) if rec_max is not None else 0.0
    if prev_canonical > 0:
        metrics["prev_token_robustness"] = max(0.0, min(1.0, prev_max / prev_canonical))
    else:
        metrics["prev_token_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical degeneracy check; skips the (expensive) jury when True.

    True only when the result is clearly degenerate: NaN/inf math, or the
    canonical previous-token mass fails to beat the uniform baseline by more
    than 10%. Never True for a borderline-but-real previous-token head.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    prev = metrics.get("prev_token_attn_canonical")
    baseline = metrics.get("uniform_baseline")
    if not _is_num(prev) or not _is_num(baseline):
        return True
    if baseline > 0 and prev <= baseline * 1.1:
        return True

    return False


GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
