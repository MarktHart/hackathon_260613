"""Benchmark for the attention_argmin goal.

Consumes the payload returned by task.evaluate and produces a flat dict of
scalar metrics. Pure Python, deterministic, side-effect free.
"""
import math

VERSION = 1

# Synthetic NumPy task — no GPU needed.
GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU


def _fmt_gap(gap: float) -> str:
    """0.5 -> '0p50', 1.0 -> '1p00', 0.05 -> '0p05'.

    Two decimals so neighbouring gaps (0.05 vs 0.1) never collide on one key.
    """
    s = f"{float(gap):.2f}"
    return s.replace(".", "p")


def _sharpness(rec: dict) -> float:
    """Concentration of attention on the argmin, relative to the uniform share.

    sharpness = attn_at_min / (1 / seq_len) = attn_at_min * seq_len

    For a normalised head this is 1.0 under uniform attention and `seq_len`
    (the maximum) for a perfect argmin head that puts all mass on the true
    minimum. Bounded in [0, seq_len] and well-defined at the optimum — a
    ratio against `attn_at_others` instead would divide by zero exactly when
    the head is perfect (no mass left on other positions).
    """
    attn_at_min = float(rec["attn_at_min"])
    seq_len = float(rec["seq_len"])
    if seq_len <= 0.0 or not math.isfinite(seq_len) or not math.isfinite(attn_at_min):
        return 0.0
    return attn_at_min * seq_len


def _require(d: dict, key: str, ctx: str):
    if key not in d:
        raise KeyError(f"payload {ctx} missing required key {key!r}")
    return d[key]


def score(payload: dict) -> dict[str, float | int]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    version = _require(payload, "version", "")
    canonical = _require(payload, "canonical", "")
    sweep = _require(payload, "sweep", "")
    baseline = _require(payload, "linear_baseline", "")

    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    baseline_canonical = _require(baseline, "canonical", "['linear_baseline']")
    baseline_sweep = _require(baseline, "sweep", "['linear_baseline']")
    if not isinstance(baseline_sweep, list) or len(baseline_sweep) != len(sweep):
        raise ValueError(
            "payload['linear_baseline']['sweep'] must match 'sweep' in length"
        )

    metrics: dict[str, float | int] = {"version": int(version)}

    # ----- canonical headline metrics -----
    metrics["argmin_sharpness_canonical"] = _sharpness(canonical)
    metrics["argmin_accuracy_canonical"] = float(canonical["argmax_correct"])
    metrics["argmin_attn_at_min_canonical"] = float(canonical["attn_at_min"])

    metrics["linear_baseline_sharpness_canonical"] = _sharpness(baseline_canonical)
    metrics["linear_baseline_accuracy_canonical"] = float(
        baseline_canonical["argmax_correct"]
    )
    metrics["lift_over_baseline_canonical"] = (
        metrics["argmin_sharpness_canonical"]
        - metrics["linear_baseline_sharpness_canonical"]
    )

    # ----- per-slice metrics -----
    sharpnesses = []
    for rec, brec in zip(sweep, baseline_sweep):
        tag = _fmt_gap(rec["gap"])
        sh = _sharpness(rec)
        bsh = _sharpness(brec)
        sharpnesses.append(sh)
        metrics[f"argmin_sharpness_gap_{tag}"] = sh
        metrics[f"argmin_accuracy_gap_{tag}"] = float(rec["argmax_correct"])
        metrics[f"linear_baseline_sharpness_gap_{tag}"] = bsh
        metrics[f"lift_over_baseline_gap_{tag}"] = sh - bsh

    # ----- robustness across the sweep -----
    hi = max(sharpnesses)
    lo = min(sharpnesses)
    metrics["argmin_robustness"] = (lo / hi) if hi > 0.0 else 0.0
    metrics["worst_slice_sharpness"] = lo

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    # Any NaN/inf math failure.
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # No lift over the no-mechanism baseline at the canonical condition.
    sharp = metrics.get("argmin_sharpness_canonical")
    base = metrics.get("linear_baseline_sharpness_canonical")
    if isinstance(sharp, (int, float)) and isinstance(base, (int, float)):
        if sharp <= base:
            return True
    return False
