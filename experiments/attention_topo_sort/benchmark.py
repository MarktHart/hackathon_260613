"""Benchmark for the `attention_topo_sort` goal.

Pure Python. No imports from any attempt directory. No I/O, no network, no
time-dependent values. Consumes the payload returned by `task.evaluate` and
returns a flat dict of named scalar metrics.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _density_key(prefix: str, density: float) -> str:
    """Slice key name, e.g. ('topo_respect_density', 0.3) -> 'topo_respect_density_0p3'."""
    return f"{prefix}_{density:g}".replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from a task.evaluate() payload.

    Expected payload keys:
        canonical_density (float), n_nodes (int), n_dags (int),
        model_name (str), sweep (list[record]).
    Each sweep record: {density, topo_respect, uniform_respect, pairs}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("canonical_density", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    canonical_density = payload["canonical_density"]
    if not isinstance(canonical_density, (int, float)):
        raise ValueError(
            f"payload['canonical_density'] must be numeric, "
            f"got {type(canonical_density).__name__}"
        )
    canonical_density = float(canonical_density)

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")

    metrics: dict[str, float | int] = {"version": VERSION}

    # Edge case: empty sweep -> degenerate-but-defined metrics.
    if len(sweep) == 0:
        metrics["topo_respect_canonical"] = 0.0
        metrics["uniform_baseline_canonical"] = 0.5
        metrics["lift_over_uniform_canonical"] = 0.0
        metrics["topo_robustness"] = 0.0
        return metrics

    canonical_respect: float | None = None
    canonical_uniform: float | None = None
    norm_skills: list[float] = []

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        for rk in ("density", "topo_respect", "uniform_respect"):
            if rk not in rec:
                raise KeyError(f"sweep record missing {rk!r}")

        density = float(rec["density"])
        respect = float(rec["topo_respect"])
        uniform = float(rec["uniform_respect"])

        metrics[_density_key("topo_respect_density", density)] = respect
        metrics[_density_key("uniform_baseline_density", density)] = uniform
        metrics[_density_key("lift_over_uniform_density", density)] = respect - uniform

        # Normalised skill of this slice: chance (0.5) -> 0, perfect (1.0) -> 1.
        norm_skills.append(max(0.0, min(1.0, (respect - 0.5) / 0.5)))

        if math.isclose(density, canonical_density, abs_tol=1e-9):
            canonical_respect = respect
            canonical_uniform = uniform

    # Canonical headline (fall back to first slice if canonical density absent).
    if canonical_respect is None:
        canonical_respect = float(sweep[0]["topo_respect"])
        canonical_uniform = float(sweep[0]["uniform_respect"])

    metrics["topo_respect_canonical"] = canonical_respect
    metrics["uniform_baseline_canonical"] = canonical_uniform
    metrics["lift_over_uniform_canonical"] = canonical_respect - canonical_uniform

    # Headline summary: worst-slice normalised skill in [0, 1].
    metrics["topo_robustness"] = float(min(norm_skills)) if norm_skills else 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are mechanically degenerate, to skip the (costly) jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that does not even beat chance (0.5) at the canonical condition
    # encodes no partial order — mechanically degenerate.
    canonical = metrics.get("topo_respect_canonical")
    if isinstance(canonical, (int, float)) and canonical <= 0.5:
        return True

    return False
