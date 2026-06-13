"""Benchmark for the attention_astar goal.

Pure Python (stdlib only). Deterministic, side-effect free. Consumes the
payload produced by task.evaluate and returns a flat dict of scalar metrics.
See README.md for the contract.
"""

import math

VERSION = 1

# Pipeline-only hook: GPU slots the attempt subprocess needs (min clamps to 1).
GPU_REQUIREMENT = 1

_SWEEP_RECORD_KEYS = (
    "obstacle_density",
    "n_grids",
    "attention_entropy",
    "heuristic_alignment",
    "top1_optimal_rate",
    "top3_optimal_rate",
    "path_optimality_gap",
    "linear_baseline_alignment",
)


def _dkey(prefix: str, density: float) -> str:
    """('astar_alignment', 0.2) -> 'astar_alignment_density_0p2'."""
    return f"{prefix}_density_{density:.1f}".replace(".", "p")


def _validate(payload: dict) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for key in ("version", "grid_size", "heuristic", "num_grids",
                "canonical_density_index", "sweep"):
        if key not in payload:
            raise KeyError(f"Missing required payload key: {key!r}")

    if not isinstance(payload["version"], int):
        raise ValueError("payload['version'] must be int")
    if payload["version"] != VERSION:
        raise ValueError(
            f"Unsupported payload version {payload['version']}; expected {VERSION}"
        )

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep record {i} must be a dict")
        for k in _SWEEP_RECORD_KEYS:
            if k not in rec:
                raise KeyError(f"sweep record {i} missing key {k!r}")
        if not isinstance(rec["n_grids"], int) or rec["n_grids"] <= 0:
            raise ValueError(f"sweep record {i}: n_grids must be a positive int")

    cidx = payload["canonical_density_index"]
    if not isinstance(cidx, int) or not (0 <= cidx < len(sweep)):
        raise ValueError(
            f"canonical_density_index {cidx!r} out of range for sweep of length {len(sweep)}"
        )


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the task.evaluate payload."""
    _validate(payload)

    sweep = payload["sweep"]
    cidx = payload["canonical_density_index"]
    canon = sweep[cidx]
    canon_density = float(canon["obstacle_density"])

    metrics: dict[str, float | int] = {"version": VERSION}

    alignments = []
    for rec in sweep:
        d = float(rec["obstacle_density"])
        align = float(rec["heuristic_alignment"])
        alignments.append(align)
        metrics[_dkey("astar_alignment", d)] = align

    # --- Canonical slice ---
    metrics["astar_alignment_canonical"] = float(canon["heuristic_alignment"])
    metrics[_dkey("astar_entropy", canon_density)] = float(canon["attention_entropy"])
    metrics["top1_optimal_canonical"] = float(canon["top1_optimal_rate"])
    metrics["top3_optimal_canonical"] = float(canon["top3_optimal_rate"])
    metrics["path_gap_canonical"] = float(canon["path_optimality_gap"])

    baseline_canonical = float(canon["linear_baseline_alignment"])
    metrics[_dkey("linear_baseline_alignment", canon_density)] = baseline_canonical
    metrics["lift_over_baseline_canonical"] = (
        metrics["astar_alignment_canonical"] - baseline_canonical
    )

    # --- Headline robustness: alignment retained at the hardest density ---
    max_align = max(alignments)
    min_align = min(alignments)
    if max_align > 1e-12:
        robustness = min_align / max_align
    else:
        robustness = 0.0
    metrics["density_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are mechanically degenerate, to skip the (expensive) jury.

    Conservative: never True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    align = metrics.get("astar_alignment_canonical")
    baseline = metrics.get("linear_baseline_alignment_density_0p2")
    if isinstance(align, (int, float)) and isinstance(baseline, (int, float)):
        # Uniform attention scores ~0 alignment; failing to beat it (or scoring
        # negative) means no A*-like structure at the canonical condition.
        if align <= baseline:
            return True

    return False
