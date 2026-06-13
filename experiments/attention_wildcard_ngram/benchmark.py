"""Benchmark for the wildcard n-gram attention-skipping goal.

Consumes the payload produced by ``task.evaluate`` (a sweep over wildcard
span, each record carrying an attention ``sharpness`` measured at the target
query position) and returns a flat dict of scalar metrics.

Direction of better: bigger is better for every metric except the explicitly
named ``linear_baseline_*`` references (which are neutral anchors).
"""

import math

VERSION = 1


def _baseline_sharpness(span: int, seq_len: int) -> float:
    """Analytic sharpness of a *uniform*-attention head under task.evaluate's
    sharpness formula:

        sharpness = mean_attn_on_anchor / (mean_attn_on_wildcards
                                           + mean_attn_on_others + 1e-8)

    With uniform attention every key receives weight ``1/seq_len`` so each of
    the three per-region means equals ``1/seq_len`` — except the wildcard mean,
    which is 0 when there are no wildcard positions (span == 0).
    """
    w = 1.0 / float(seq_len)
    mean_anchor = w
    mean_wild = w if span > 0 else 0.0
    mean_others = w  # there is always at least one filler/other position
    return mean_anchor / (mean_wild + mean_others + 1e-8)


def score(payload: dict) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload)!r}")

    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark "
            f"VERSION {VERSION}"
        )

    sweep = payload.get("sweep")
    if not isinstance(sweep, list):
        raise KeyError("payload missing list key 'sweep'")
    if len(sweep) == 0:
        raise ValueError("payload 'sweep' is empty")

    canonical_span = int(payload.get("canonical_span", 1))
    seq_len = int(payload.get("seq_len", 16))
    if seq_len <= 1:
        raise ValueError(f"seq_len must be > 1, got {seq_len}")

    metrics: dict[str, float | int] = {"version": VERSION}

    sharp_by_span: dict[int, float] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError(f"sweep record must be a dict, got {type(rec)!r}")
        for key in ("wildcard_span", "sharpness"):
            if key not in rec:
                raise KeyError(f"sweep record missing key {key!r}: {rec!r}")
        span = int(rec["wildcard_span"])
        sharpness = float(rec["sharpness"])
        sharp_by_span[span] = sharpness

        metrics[f"sharpness_wildcard_span_{span}"] = sharpness
        metrics[f"linear_baseline_sharpness_wildcard_span_{span}"] = (
            _baseline_sharpness(span, seq_len)
        )

    if canonical_span not in sharp_by_span:
        raise ValueError(
            f"canonical_span {canonical_span} not present in sweep spans "
            f"{sorted(sharp_by_span)}"
        )

    canon = sharp_by_span[canonical_span]
    canon_baseline = _baseline_sharpness(canonical_span, seq_len)

    metrics["sharpness_canonical"] = canon
    metrics["lift_over_baseline_canonical"] = canon - canon_baseline

    # Headline: how well sharpness holds when a wildcard is inserted between
    # anchor and target, relative to the no-wildcard (span 0) condition.
    base = sharp_by_span.get(0)
    if base is not None and base > 0.0:
        metrics["wildcard_skip_robustness"] = canon / base
    else:
        metrics["wildcard_skip_robustness"] = 0.0

    # Mean sharpness across the full span sweep (context for the headline).
    if sharp_by_span:
        metrics["mean_sharpness"] = sum(sharp_by_span.values()) / len(sharp_by_span)
    else:
        metrics["mean_sharpness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Short-circuit the jury on mechanically-detectable failures."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    robustness = metrics.get("wildcard_skip_robustness")
    if isinstance(robustness, (int, float)) and robustness < 0.5:
        return True

    canon = metrics.get("sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_wildcard_span_1")
    if isinstance(canon, (int, float)) and isinstance(baseline, (int, float)):
        if canon <= baseline:
            return True

    return False


GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
