"""
Benchmark for the attention-deduplication goal.

Consumes the payload produced by task.evaluate(model_fn) and emits flat scalar
metrics. Pure Python, deterministic, side-effect free, defensive on inputs.

Headline metric: `dedup_robustness` — mean attention mass placed on the
previous occurrence of duplicate tokens, averaged across the dup-rate sweep,
in [0, 1]. Bigger is better. Every metric in this file is bigger-is-better.
"""

import math

VERSION = 1

# Pipeline-only hook: number of GPU slots the *attempt* subprocess needs.
GPU_REQUIREMENT = 1


def _fmt(x: float) -> str:
    """Slice-key form for a float, e.g. 0.5 -> '0p5', 0.1 -> '0p1'."""
    return f"{x:.1f}".replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the task.evaluate payload.

    Expected payload keys:
        version (int == VERSION), canonical_dup_rate (float),
        dup_rates (list[float]), sweep (list[record]).
    Each sweep record:
        {dup_rate, n_dup_positions, n_first_seen, dedup_mass, dedup_accuracy,
         first_seen_self_mass, baseline_dedup_mass, baseline_dedup_accuracy}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_dup_rate", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    canonical = float(payload["canonical_dup_rate"])

    metrics: dict[str, float | int] = {"version": VERSION}

    masses: list[float] = []
    accs: list[float] = []
    canon_rec = None

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        if "dup_rate" not in rec:
            raise KeyError("sweep record missing 'dup_rate'")

        r = float(rec["dup_rate"])
        key = _fmt(r)
        dmass = float(rec.get("dedup_mass", 0.0))
        dacc = float(rec.get("dedup_accuracy", 0.0))
        bmass = float(rec.get("baseline_dedup_mass", 0.0))

        metrics[f"dedup_mass_rate_{key}"] = dmass
        metrics[f"dedup_accuracy_rate_{key}"] = dacc
        metrics[f"baseline_dedup_mass_rate_{key}"] = bmass

        masses.append(dmass)
        accs.append(dacc)
        if abs(r - canonical) < 1e-9:
            canon_rec = rec

    # Fallback: if the declared canonical rate isn't present, use the middle slice.
    if canon_rec is None:
        canon_rec = sweep[len(sweep) // 2]

    canon_mass = float(canon_rec.get("dedup_mass", 0.0))
    canon_base = float(canon_rec.get("baseline_dedup_mass", 0.0))

    metrics["dedup_mass_canonical"] = canon_mass
    metrics["dedup_accuracy_canonical"] = float(canon_rec.get("dedup_accuracy", 0.0))
    metrics["first_seen_self_mass_canonical"] = float(
        canon_rec.get("first_seen_self_mass", 0.0)
    )
    metrics["uniform_baseline_dedup_mass_canonical"] = canon_base
    metrics["lift_over_uniform_canonical"] = canon_mass - canon_base

    # Headline: mean dedup mass across the whole sweep, clamped to [0, 1].
    robustness = sum(masses) / len(masses) if masses else 0.0
    metrics["dedup_robustness"] = float(max(0.0, min(1.0, robustness)))
    metrics["dedup_accuracy_mean"] = float(sum(accs) / len(accs)) if accs else 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    mass = metrics.get("dedup_mass_canonical")
    base = metrics.get("uniform_baseline_dedup_mass_canonical")
    if isinstance(mass, (int, float)) and isinstance(base, (int, float)):
        # Failing to beat a uniform-causal attention at the canonical condition
        # means no deduplication mechanism is present.
        if mass <= base:
            return True

    return False
