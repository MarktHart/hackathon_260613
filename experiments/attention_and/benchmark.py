"""Benchmark for the `attention_and` goal.

VERSION 2 extends the metric from a single orthogonal measurement to a sweep
across `cos(q_A, q_B)`, so attempts are also judged on whether they survive
under superposition (non-orthogonal concept directions), not just at perfect
orthogonality.

## Payload contract

    sweep: list[dict]
        One record per cosine slice:
            cosine: float                      cos(q_A, q_B) for this slice
            softmax_weights: dict[str, float]  per-token softmax mass
            linear_weights:  dict[str, float]  per-token linear-baseline mass
        Each weights dict must sum to ~1 and contain every label used below.
    both_label: str                            which key in weights is the AND-target
    single_feature_labels: list[str]           A-only and B-only token labels
    canonical_scale: float                     scale used for the whole sweep (record-keeping)

## Metrics

Per slice (one named scalar per cosine value `c`):
    and_sharpness_cos_<c>              softmax[both] / mean(softmax[single])
    linear_baseline_sharpness_cos_<c>  same ratio for the linear baseline (no-`exp` ceiling)

Summary (the headline values):
    superposition_robustness   and_sharpness at highest cosine / at lowest cosine
                               (1.0 = method holds up perfectly; → 0 = collapses)
    and_sharpness_canonical    and_sharpness at the lowest cosine (most orthogonal)
    softmax_both_mass_canonical softmax[both] at the lowest cosine
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 2

# Optional pipeline hooks read by `agentic.pipeline`. The synthetic soft-AND
# experiment is CPU-bound numpy — it does not need a GPU at all. We still
# request 1 slot so the pool serialises with other goals, but a future
# attention_and variant that loaded a real transformer could bump this.
GPU_REQUIREMENT: int = 1


def is_obviously_broken(metrics: dict[str, Any]) -> bool:
    """Skip the (expensive) jury if the run is degenerate.

    Broken means:
    - Any metric is NaN or ±inf (math failure in the attempt).
    - `and_sharpness_canonical` doesn't meaningfully beat the linear baseline —
      the attempt didn't even produce AND-gating, so there is nothing to grade.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    sharp = metrics.get("and_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_cos_0p0")
    if isinstance(sharp, int | float) and isinstance(baseline, int | float):
        if sharp <= baseline * 1.5:
            return True
    return False


def _slice_key(cosine: float) -> str:
    """Stable key for a per-slice metric. 0.7 -> 'cos_0p7'."""
    return f"cos_{str(round(cosine, 3)).replace('.', 'p')}"


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    sweep: list[dict[str, Any]] = payload["sweep"]
    both: str = payload["both_label"]
    singles: list[str] = payload["single_feature_labels"]

    if not sweep:
        raise ValueError("sweep must be non-empty.")
    if not singles:
        raise ValueError("single_feature_labels must be non-empty.")

    metrics: dict[str, float | int] = {"version": VERSION}
    sharpness_by_cosine: dict[float, float] = {}

    for slice_ in sweep:
        cosine = float(slice_["cosine"])
        sw: dict[str, float] = slice_["softmax_weights"]
        lw: dict[str, float] = slice_["linear_weights"]

        mean_single_softmax = sum(sw[t] for t in singles) / len(singles)
        mean_single_linear = sum(lw[t] for t in singles) / len(singles)

        softmax_sharpness = (
            sw[both] / mean_single_softmax if mean_single_softmax > 0 else float("inf")
        )
        linear_sharpness = (
            lw[both] / mean_single_linear if mean_single_linear > 0 else float("inf")
        )

        suffix = _slice_key(cosine)
        metrics[f"and_sharpness_{suffix}"] = float(softmax_sharpness)
        metrics[f"linear_baseline_sharpness_{suffix}"] = float(linear_sharpness)
        sharpness_by_cosine[cosine] = float(softmax_sharpness)

    sorted_cosines = sorted(sharpness_by_cosine)
    low_cos, high_cos = sorted_cosines[0], sorted_cosines[-1]
    base = sharpness_by_cosine[low_cos]
    high = sharpness_by_cosine[high_cos]
    canonical_slice = next(s for s in sweep if float(s["cosine"]) == low_cos)

    metrics["superposition_robustness"] = (
        float(high / base) if base > 0 else float("inf")
    )
    metrics["and_sharpness_canonical"] = float(base)
    metrics["softmax_both_mass_canonical"] = float(
        canonical_slice["softmax_weights"][both]
    )

    return metrics
