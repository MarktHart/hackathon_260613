import math
from typing import Any, Dict

VERSION = 1

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU

_SEGMENTS = ("segA", "segB")


def _mean(xs) -> float:
    xs = list(xs)
    if not xs:
        return 0.0
    return sum(float(x) for x in xs) / len(xs)


def _mean_sharpness(seg: Dict[str, Any]) -> float:
    return _mean(seg.get("head_sharpness", []))


def _get_sweep_segment(sweep, segment: str) -> Dict[str, Any]:
    for rec in sweep:
        if isinstance(rec, dict) and rec.get("query_segment") == segment:
            return rec
    raise KeyError(f"segment {segment!r} not found in sweep")


def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    """
    Compute metrics from the payload returned by task.evaluate().
    Returns a flat dict of scalar metrics.
    """
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    for k in ("version", "config", "sweep", "linear_baseline"):
        if k not in payload:
            raise KeyError(f"missing required payload key: {k}")

    if payload["version"] != VERSION:
        raise ValueError(f"payload version {payload['version']} != benchmark VERSION {VERSION}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) != len(_SEGMENTS):
        raise ValueError(f"sweep must be a list of {len(_SEGMENTS)} records (segA, segB)")

    linear_baseline = payload["linear_baseline"]
    if not isinstance(linear_baseline, dict) or any(s not in linear_baseline for s in _SEGMENTS):
        raise KeyError("linear_baseline must contain 'segA' and 'segB'")

    seg = {s: _get_sweep_segment(sweep, s) for s in _SEGMENTS}
    base = {s: linear_baseline[s] for s in _SEGMENTS}

    # Validate finite numeric fields per record.
    for s in _SEGMENTS:
        rec = seg[s]
        for k in ("within_seg_attn", "delim_attn", "cross_seg_attn", "eos_attn"):
            v = rec.get(k)
            if not isinstance(v, (int, float)) or math.isnan(float(v)) or math.isinf(float(v)):
                raise ValueError(f"sweep segment {s!r} field {k!r} must be finite, got {v!r}")
        hs = rec.get("head_sharpness")
        if not isinstance(hs, list) or not hs:
            raise ValueError(f"sweep segment {s!r} must have a non-empty 'head_sharpness' list")

    # Per-slice sharpness (within - best competing region), averaged over heads.
    sharp = {s: _mean_sharpness(seg[s]) for s in _SEGMENTS}
    base_sharp = {s: _mean_sharpness(base[s]) for s in _SEGMENTS}

    sharpness_canonical = _mean(sharp.values())
    base_sharpness_canonical = _mean(base_sharp.values())

    cross = {s: float(seg[s].get("cross_seg_attn", 0.0)) for s in _SEGMENTS}
    delim = {s: float(seg[s].get("delim_attn", 0.0)) for s in _SEGMENTS}

    metrics: Dict[str, float | int] = {
        "version": VERSION,
        # Headline: how sharply heads respect the segment boundary.
        "boundary_sharpness_canonical": sharpness_canonical,
        "boundary_sharpness_segA": sharp["segA"],
        "boundary_sharpness_segB": sharp["segB"],
        # Boundary crossing: attention mass leaking to the other segment.
        "boundary_crossing_rate_canonical": _mean(cross.values()),
        "boundary_crossing_rate_segA": cross["segA"],
        "boundary_crossing_rate_segB": cross["segB"],
        # Delimiter leakage: attention mass parked on the delimiter token.
        "delim_leakage_canonical": _mean(delim.values()),
        "delim_leakage_segA": delim["segA"],
        "delim_leakage_segB": delim["segB"],
        # Reference baseline (uniform attention, no boundary mechanism).
        "linear_baseline_sharpness_canonical": base_sharpness_canonical,
        "linear_baseline_sharpness_segA": base_sharp["segA"],
        "linear_baseline_sharpness_segB": base_sharp["segB"],
        "lift_over_linear_sharpness": sharpness_canonical - base_sharpness_canonical,
    }
    return metrics


def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    """
    Pipeline hook: True only for mechanically degenerate attempts (skips jury).
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    sharp = metrics.get("boundary_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_canonical")
    if not isinstance(sharp, (int, float)) or not isinstance(baseline, (int, float)):
        return True

    # Uniform baseline has sharpness 0, so require a clear positive margin.
    if sharp <= max(baseline, 0.0) + 0.05:
        return True

    # More mass to the other segment than a uniform read would give is broken.
    cross = metrics.get("boundary_crossing_rate_canonical")
    if isinstance(cross, (int, float)) and cross > 0.5:
        return True

    return False
