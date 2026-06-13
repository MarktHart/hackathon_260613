"""
Benchmark scoring for the `attention_viterbi` goal.

Consumes the payload returned by task.evaluate() and produces a flat dict of
named scalar metrics. Pure Python; deterministic; side-effect free.
"""

from __future__ import annotations

import math

VERSION = 1

# How many GPU slots the attempt subprocess needs (attention-only 2L model).
GPU_REQUIREMENT = 1


def _require(payload: dict, key: str):
    if key not in payload:
        raise KeyError(f"Missing required payload key: {key!r}")
    return payload[key]


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics.

    Expected payload keys:
        version (int == VERSION), n_layers (int), n_heads (int),
        per_head (list of {layer, head, excess}),
        positional (list of {pos, excess, n}),
        best_head ({layer, head}),
        baseline_uniform_excess (float), baseline_random_excess (float).

    All `excess` values are mean excess attention on the Viterbi predecessor,
    bounded in (-1, 1); bigger is better. Uniform causal attention scores 0.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    version = _require(payload, "version")
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version {version}; expected {VERSION}.")

    n_layers = int(_require(payload, "n_layers"))
    n_heads = int(_require(payload, "n_heads"))
    per_head = _require(payload, "per_head")
    positional = _require(payload, "positional")
    baseline_uniform = float(_require(payload, "baseline_uniform_excess"))
    baseline_random = float(_require(payload, "baseline_random_excess"))

    if not isinstance(per_head, (list, tuple)) or len(per_head) != n_layers * n_heads:
        raise ValueError(
            f"per_head must be a list of length n_layers*n_heads "
            f"({n_layers * n_heads}); got {len(per_head) if isinstance(per_head, (list, tuple)) else type(per_head).__name__}"
        )
    if not isinstance(positional, (list, tuple)):
        raise ValueError("positional must be a list")

    metrics: dict[str, float | int] = {"version": VERSION}

    # --- Per-head slice metrics ---
    head_excess: dict[tuple[int, int], float] = {}
    excess_vals: list[float] = []
    for rec in per_head:
        if not isinstance(rec, dict):
            raise ValueError("each per_head record must be a dict")
        layer = int(rec.get("layer", -1))
        head = int(rec.get("head", -1))
        ex = float(rec.get("excess", 0.0))
        head_excess[(layer, head)] = ex
        excess_vals.append(ex)
        metrics[f"viterbi_attention_layer_{layer}_head_{head}"] = ex

    if not excess_vals:
        raise ValueError("per_head contained no usable records")

    # --- Headline + aggregates ---
    canonical = max(excess_vals)               # strongest Viterbi head
    metrics["viterbi_attention_canonical"] = float(canonical)
    metrics["viterbi_attention_mean"] = float(sum(excess_vals) / len(excess_vals))

    best_head = payload.get("best_head", {})
    metrics["best_head_layer"] = int(best_head.get("layer", -1))
    metrics["best_head_head"] = int(best_head.get("head", -1))

    # --- Baselines + lift ---
    metrics["baseline_uniform_excess"] = baseline_uniform
    metrics["baseline_random_excess"] = baseline_random
    metrics["linear_baseline_viterbi_attention"] = baseline_uniform
    metrics["lift_over_uniform"] = float(canonical - baseline_uniform)
    metrics["lift_over_random"] = float(canonical - baseline_random)

    # --- Per-position slice metrics + robustness ---
    n_positive = 0
    n_pos = 0
    for rec in positional:
        if not isinstance(rec, dict):
            raise ValueError("each positional record must be a dict")
        pos = int(rec.get("pos", -1))
        ex = float(rec.get("excess", 0.0))
        metrics[f"viterbi_attention_pos_{pos}"] = ex
        n_pos += 1
        if ex > 0.0:
            n_positive += 1

    # Fraction of query positions where the best head beats uniform reading.
    # In [0, 1]; measures how consistently the Viterbi structure holds.
    if n_pos > 0:
        metrics["viterbi_robustness"] = float(n_positive / n_pos)
    else:
        metrics["viterbi_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    True iff metrics are mechanically degenerate, so the pipeline can skip the
    jury. Never True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    canonical = metrics.get("viterbi_attention_canonical")
    uniform = metrics.get("baseline_uniform_excess")
    # Uniform causal attention has excess exactly 0. A real Viterbi head must
    # put *more* mass on the predecessor than a uniform reader; if even the best
    # head fails to beat the uniform baseline there is no mechanism to judge.
    if isinstance(canonical, (int, float)) and isinstance(uniform, (int, float)):
        if canonical <= uniform:
            return True

    return False
