"""
benchmark.py for attention_mst.

Consumes the payload produced by task.evaluate() and emits a flat dict of
scalar metrics. Pure Python, deterministic, side-effect free, defensive.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs on the GPU; minimum is 1.
GPU_REQUIREMENT = 1


def _noise_key(prefix: str, noise: float) -> str:
    """Slice key, e.g. ('edge_f1', 0.5) -> 'edge_f1_noise_0p5'."""
    return f"{prefix}_noise_{noise:.1f}".replace(".", "p")


def _index_by_noise(records, what):
    out = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"each {what} record must be a dict, got {type(rec).__name__}")
        if "noise_level" not in rec:
            raise KeyError(f"{what} record missing 'noise_level'")
        out[round(float(rec["noise_level"]), 6)] = rec
    return out


def score(payload: dict) -> dict[str, float | int]:
    """
    Expected payload keys:
        version (int == VERSION), model_name (str), n_heads (int),
        canonical_noise (float), noise_levels (list[float]),
        sweep (list[record]), baseline (list[record]).

    Each sweep / baseline record:
        {noise_level, edge_f1, precision, recall, auroc, auprc,
         weight_ratio, n_seeds}.

    Returns a flat dict of scalar metrics:
        version,
        edge_f1_noise_<v>, auroc_noise_<v>, weight_ratio_noise_<v>,
        baseline_edge_f1_noise_<v>, baseline_auroc_noise_<v>,
        edge_f1_canonical, auroc_canonical,
        baseline_edge_f1_canonical, lift_over_baseline_canonical,
        mst_recovery (headline: mean edge_f1 across the sweep),
        mst_recovery_robustness (edge_f1 retained at max vs min noise).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "n_heads", "canonical_noise",
                "noise_levels", "sweep", "baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"unsupported payload version {version}; expected {VERSION}")

    noise_levels = payload["noise_levels"]
    if not isinstance(noise_levels, (list, tuple)) or len(noise_levels) == 0:
        raise ValueError("payload['noise_levels'] must be a non-empty list")
    noise_levels = [float(s) for s in noise_levels]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(noise_levels):
        raise ValueError("payload['sweep'] must be a list the same length as noise_levels")

    baseline = payload["baseline"]
    if not isinstance(baseline, (list, tuple)) or len(baseline) != len(noise_levels):
        raise ValueError("payload['baseline'] must be a list the same length as noise_levels")

    sweep_by_noise = _index_by_noise(sweep, "sweep")
    base_by_noise = _index_by_noise(baseline, "baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    method_f1: list[float] = []
    for noise in noise_levels:
        key = round(noise, 6)
        srec = sweep_by_noise.get(key, {})
        brec = base_by_noise.get(key, {})

        m_f1 = float(srec.get("edge_f1", 0.0))
        m_auroc = float(srec.get("auroc", 0.0))
        m_wratio = float(srec.get("weight_ratio", 0.0))
        b_f1 = float(brec.get("edge_f1", 0.0))
        b_auroc = float(brec.get("auroc", 0.0))

        metrics[_noise_key("edge_f1", noise)] = m_f1
        metrics[_noise_key("auroc", noise)] = m_auroc
        metrics[_noise_key("weight_ratio", noise)] = m_wratio
        metrics[_noise_key("baseline_edge_f1", noise)] = b_f1
        metrics[_noise_key("baseline_auroc", noise)] = b_auroc

        method_f1.append(m_f1)

    # --- Canonical condition ---
    canonical_noise = float(payload["canonical_noise"])
    metrics["edge_f1_canonical"] = float(
        metrics.get(_noise_key("edge_f1", canonical_noise), 0.0)
    )
    metrics["auroc_canonical"] = float(
        metrics.get(_noise_key("auroc", canonical_noise), 0.0)
    )
    base_canonical = float(
        metrics.get(_noise_key("baseline_edge_f1", canonical_noise), 0.0)
    )
    metrics["baseline_edge_f1_canonical"] = base_canonical
    metrics["lift_over_baseline_canonical"] = (
        metrics["edge_f1_canonical"] - base_canonical
    )

    # --- Headline: mean edge_f1 across the sweep ---
    metrics["mst_recovery"] = (
        float(sum(method_f1) / len(method_f1)) if method_f1 else 0.0
    )

    # --- Robustness: edge_f1 retained at max noise vs min noise, in [0, 1] ---
    f1_low = float(metrics.get(_noise_key("edge_f1", noise_levels[0]), 0.0))
    f1_high = float(metrics.get(_noise_key("edge_f1", noise_levels[-1]), 0.0))
    if f1_low > 1e-12:
        robustness = f1_high / f1_low
    else:
        robustness = 0.0
    metrics["mst_recovery_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    True iff metrics are mechanically degenerate, so the pipeline can skip the
    jury. Never True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that does not even beat the no-mechanism baseline at the
    # canonical condition is not worth jurying.
    method = metrics.get("edge_f1_canonical")
    base = metrics.get("baseline_edge_f1_canonical")
    if isinstance(method, (int, float)) and isinstance(base, (int, float)):
        if method <= base:
            return True

    return False
