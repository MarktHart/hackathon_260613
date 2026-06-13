"""Benchmark for the attention_lcs goal.

Pure Python. Deterministic. No I/O, no imports from any attempt directory.
Consumes the payload returned by task.evaluate and returns flat scalar metrics.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs on the GPU.
GPU_REQUIREMENT = 1


def _head_key(prefix: str, head: int) -> str:
    return f"{prefix}_head_{int(head)}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from task.evaluate()'s payload.

    Expected payload:
        version (int == VERSION)
        config (dict)
        random_baseline_mass (float)
        sweep (list of per-head records, each:
            {head: int, lcs_attention_mass: float, lcs_lift: float,
             n_query_positions: int})
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "random_baseline_mass", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version {version}; expected {VERSION}.")

    baseline = payload["random_baseline_mass"]
    if not isinstance(baseline, (int, float)) or not math.isfinite(float(baseline)):
        raise ValueError(f"random_baseline_mass must be a finite number, got {baseline!r}")
    baseline = float(baseline)

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")
    if len(sweep) == 0:
        raise ValueError("payload['sweep'] must contain at least one head record")

    metrics: dict[str, float | int] = {"version": VERSION}
    metrics["random_baseline_mass"] = baseline

    mass_vals: list[float] = []
    lift_vals: list[float] = []

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        if "head" not in rec or "lcs_attention_mass" not in rec:
            raise KeyError("sweep record needs 'head' and 'lcs_attention_mass'")
        head = int(rec["head"])
        mass = float(rec["lcs_attention_mass"])
        # lcs_lift is derivable; recompute defensively so score() is robust to a
        # payload that omitted or mis-set it.
        lift = float(rec.get("lcs_lift", mass - baseline))

        metrics[_head_key("lcs_attention", head)] = mass
        metrics[_head_key("lcs_lift", head)] = lift
        mass_vals.append(mass)
        lift_vals.append(lift)

    # Headline + canonical summaries: the single best head.
    metrics["lcs_attention_canonical"] = max(mass_vals)
    metrics["lcs_lift_canonical"] = max(lift_vals)

    # Robustness: lift as a fraction of the achievable headroom above chance.
    headroom = 1.0 - baseline
    if headroom > 1e-9:
        robustness = metrics["lcs_lift_canonical"] / headroom
    else:
        robustness = 0.0
    metrics["lcs_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical degeneracy check; skips the jury when True. Never True for a
    borderline-but-real result."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    lift = metrics.get("lcs_lift_canonical")
    if isinstance(lift, (int, float)) and lift <= 0:
        # Best head does no better than uniform attention.
        return True
    return False
