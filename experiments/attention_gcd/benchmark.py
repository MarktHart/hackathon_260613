"""Benchmark for the `attention_gcd` goal.

Consumes the payload produced by ``task.evaluate`` and returns a flat dict of
named scalar metrics. Pure Python, deterministic, side-effect free.

Headline: ``gcd_decodability`` — best-layer test R² of a linear probe that
predicts gcd(a, b) from the residual stream at the SEP position. Bigger is
better; a value near 0 means gcd is not linearly recoverable.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs on the GPU.
GPU_REQUIREMENT = 1


def _slice_key(prefix: str, name: str) -> str:
    return f"{prefix}_{name}"


def score(payload: dict) -> dict[str, float | int]:
    # --- Input validation ---------------------------------------------------
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "attn_corr", "baseline_attn_corr", "global", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}"
        )
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    glob = payload["global"]
    if not isinstance(glob, dict):
        raise ValueError("payload['global'] must be a dict")
    for k in ("resid_r2", "resid_acc", "baseline_r2", "baseline_acc"):
        if k not in glob:
            raise KeyError(f"Missing required payload['global'] key: {k!r}")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")

    attn_corr = payload["attn_corr"]
    if not isinstance(attn_corr, (list, tuple)):
        raise ValueError("payload['attn_corr'] must be a list")

    metrics: dict[str, float | int] = {"version": VERSION}

    def _clip01(x: float) -> float:
        return float(max(0.0, min(1.0, x)))

    def _maxf(vals, default=0.0) -> float:
        vals = [float(v) for v in vals]
        return max(vals) if vals else float(default)

    # --- Attention metrics --------------------------------------------------
    all_abs_corr = []
    for row in attn_corr:
        for v in row:
            all_abs_corr.append(abs(float(v)))
    metrics["gcd_attn_corr_canonical"] = _maxf(all_abs_corr, 0.0)
    metrics["linear_baseline_attn_corr_canonical"] = abs(
        float(payload["baseline_attn_corr"])
    )
    metrics["attn_corr_lift_over_baseline"] = (
        metrics["gcd_attn_corr_canonical"]
        - metrics["linear_baseline_attn_corr_canonical"]
    )

    # --- Residual probe metrics (canonical = best layer) --------------------
    resid_r2 = list(glob["resid_r2"])
    resid_acc = list(glob["resid_acc"])
    base_r2 = float(glob["baseline_r2"])
    base_acc = float(glob["baseline_acc"])

    metrics["gcd_resid_r2_canonical"] = _clip01(_maxf(resid_r2, 0.0))
    metrics["gcd_decode_acc_canonical"] = _clip01(_maxf(resid_acc, 0.0))
    metrics["linear_baseline_resid_r2_canonical"] = _clip01(base_r2)
    metrics["linear_baseline_acc_canonical"] = _clip01(base_acc)

    # Headline summary metric.
    metrics["gcd_decodability"] = metrics["gcd_resid_r2_canonical"]

    metrics["resid_r2_lift_over_baseline"] = (
        metrics["gcd_resid_r2_canonical"]
        - metrics["linear_baseline_resid_r2_canonical"]
    )
    metrics["decode_acc_lift_over_baseline"] = (
        metrics["gcd_decode_acc_canonical"]
        - metrics["linear_baseline_acc_canonical"]
    )

    # --- Per-slice metrics over gcd bins ------------------------------------
    per_bin_acc = []
    for rec in sweep:
        if not isinstance(rec, dict) or "bin" not in rec:
            raise ValueError("each sweep record must be a dict with a 'bin' key")
        name = str(rec["bin"])
        count = int(rec.get("count", 0))
        bin_acc = _clip01(_maxf(rec.get("resid_acc", []), 0.0))
        bin_base = _clip01(float(rec.get("baseline_acc", 0.0)))
        metrics[_slice_key("gcd_decode_acc", name)] = bin_acc
        metrics[_slice_key("gcd_decode_acc_baseline", name)] = bin_base
        if count > 0:
            per_bin_acc.append(bin_acc)

    # --- Robustness: stability of decode accuracy across populated bins -----
    if per_bin_acc:
        mx = max(per_bin_acc)
        mn = min(per_bin_acc)
        metrics["gcd_decode_robustness"] = _clip01(mn / mx) if mx > 1e-12 else 0.0
    else:
        metrics["gcd_decode_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Skip the jury on mechanically degenerate results. Never True for a
    borderline-but-real result."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # The whole point is to beat a linear probe on the raw inputs. If the
    # residual probe does not exceed the raw-input baseline R², there is no
    # mechanism to interpret.
    method = metrics.get("gcd_resid_r2_canonical")
    baseline = metrics.get("linear_baseline_resid_r2_canonical")
    if isinstance(method, (int, float)) and isinstance(baseline, (int, float)):
        if method <= baseline:
            return True

    return False
