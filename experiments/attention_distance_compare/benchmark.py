"""
Benchmark for `attention_distance_compare`.

Consumes the payload produced by `task.evaluate` (mean attention weight per
positional-distance bin, globally and per layer/head, plus a uniform-attention
baseline) and reduces it to a flat dict of scalar metrics.

All metrics are bigger-is-better:
    - a steeper distance decay (more positive `*_decay_slope`) means the model
      concentrates attention on nearby keys;
    - a larger `local_attention_fraction` means more mass within distance <= 4;
    - lower `attn_entropy_canonical` (peaked) is the *interesting* regime, but
      we report it as-is and do not put it on the headline.
"""

from __future__ import annotations

import math

VERSION: int = 1
GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU

_EPS = 1e-12
# Bin centers <= 4.5 are the singleton distance-0..4 bins; token distances <= 4
# count as "local" (the distance-4 bin [4,5) has center 4.5 and is included).
_LOCAL_THRESHOLD = 4.5

_REQUIRED_KEYS = (
    "version",
    "distance_bins",
    "mean_attn_per_bin",
    "uniform_baseline_per_bin",
    "mean_attn_per_layer_head_bin",
)


def _fmt_dist(d: int) -> str:
    """Format a distance for a metric-key suffix (ints stay int-like)."""
    return str(int(d))


def _decay_slope(distance_bins: list, bin_values: list) -> float:
    """
    Fit ln(mean_attn) against log2(distance) over the *positive* distance bins
    (distance >= 1) and return the **negated** slope, so that a stronger decay
    (attention falling off with distance) yields a larger positive number.

    Returns 0.0 if fewer than two usable points exist.
    """
    xs, ys = [], []
    for d, v in zip(distance_bins, bin_values):
        if d < 1:
            continue  # skip distance 0 (self / adjacent dominates trivially)
        xs.append(math.log2(float(d)))
        ys.append(math.log(max(float(v), _EPS)))
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= _EPS:
        return 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    return float(-slope)


def _local_fraction(distance_bins: list, bin_values: list) -> float:
    total = sum(max(float(v), 0.0) for v in bin_values)
    if total <= _EPS:
        return 0.0
    local = sum(
        max(float(v), 0.0)
        for d, v in zip(distance_bins, bin_values)
        if d <= _LOCAL_THRESHOLD
    )
    return float(local / total)


def _entropy_bits(bin_values: list) -> float:
    """Shannon entropy (bits) of the bin values treated as a distribution."""
    vals = [max(float(v), 0.0) for v in bin_values]
    total = sum(vals)
    if total <= _EPS:
        return 0.0
    h = 0.0
    for v in vals:
        p = v / total
        if p > _EPS:
            h -= p * math.log2(p)
    return float(h)


def score(payload: dict) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload)!r}")
    for k in _REQUIRED_KEYS:
        if k not in payload:
            raise KeyError(f"payload missing required key {k!r}")

    distance_bins = list(payload["distance_bins"])
    mean_per_bin = list(payload["mean_attn_per_bin"])
    baseline_per_bin = list(payload["uniform_baseline_per_bin"])
    per_lh = payload["mean_attn_per_layer_head_bin"]

    n_bins = len(distance_bins)
    if n_bins == 0:
        raise ValueError("distance_bins is empty; nothing to score")
    if len(mean_per_bin) != n_bins:
        raise ValueError(
            f"mean_attn_per_bin has length {len(mean_per_bin)}, "
            f"expected {n_bins}"
        )
    if len(baseline_per_bin) != n_bins:
        raise ValueError(
            f"uniform_baseline_per_bin has length {len(baseline_per_bin)}, "
            f"expected {n_bins}"
        )

    metrics: dict[str, float | int] = {"version": VERSION}

    # Headline + supporting global metrics.
    headline = _decay_slope(distance_bins, mean_per_bin)
    baseline_slope = _decay_slope(distance_bins, baseline_per_bin)
    metrics["distance_decay_slope_canonical"] = headline
    metrics["uniform_baseline_decay_slope"] = baseline_slope
    metrics["lift_over_uniform_decay_slope"] = float(headline - baseline_slope)
    metrics["local_attention_fraction_canonical"] = _local_fraction(
        distance_bins, mean_per_bin
    )
    metrics["attn_entropy_canonical"] = _entropy_bits(mean_per_bin)

    # Per-slice: mean attention at each distance bin.
    for d, v in zip(distance_bins, mean_per_bin):
        metrics[f"mean_attn_dist_{_fmt_dist(d)}"] = float(v)

    # Per-layer-per-head decay slopes.
    if isinstance(per_lh, list):
        for li, layer in enumerate(per_lh):
            if not isinstance(layer, list):
                continue
            for hi, head_bins in enumerate(layer):
                if not isinstance(head_bins, list) or len(head_bins) != n_bins:
                    continue
                metrics[f"layer_head_decay_slope_layer{li}_head{hi}"] = (
                    _decay_slope(distance_bins, head_bins)
                )

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    # Any NaN/inf is a hard failure.
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    frac = metrics.get("local_attention_fraction_canonical")
    if isinstance(frac, (int, float)) and not (0.0 - _EPS <= frac <= 1.0 + _EPS):
        return True

    headline = metrics.get("distance_decay_slope_canonical")
    baseline = metrics.get("uniform_baseline_decay_slope")
    if isinstance(headline, (int, float)) and isinstance(baseline, (int, float)):
        # No distance mechanism beyond the uniform baseline → degenerate.
        if headline <= baseline + 1e-6:
            return True

    return False
