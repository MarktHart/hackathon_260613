"""Benchmark for attention_graph_color.

Consumes the payload produced by ``task.evaluate`` and returns a flat dict of
scalar metrics. Pure, deterministic, side-effect free.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _separation(rec: dict) -> float:
    return float(rec.get("diff_color_attention", 0.0)) - float(
        rec.get("same_color_attention", 0.0)
    )


def _edge_respect(rec: dict) -> float:
    return float(rec.get("cross_edge_diff_color", 0.0)) - float(
        rec.get("cross_edge_same_color", 0.0)
    )


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the task.evaluate payload.

    Expected payload keys:
        version (int == 1), canonical_n (int), canonical_p (float),
        n_values (list[int]), num_graphs (int), sweep (list[record]),
        baseline_sweep (list[record]).

    Each sweep record: {graph_idx, num_nodes, num_colors, edge_density,
        same_color_attention, diff_color_attention, cross_edge_same_color,
        cross_edge_diff_color, isolated_node_fraction}.
    Each baseline_sweep record: {graph_idx, num_nodes, same_color_attention,
        diff_color_attention, cross_edge_same_color, cross_edge_diff_color}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_n", "n_values", "sweep", "baseline_sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")
    baseline_sweep = payload["baseline_sweep"]
    if not isinstance(baseline_sweep, (list, tuple)):
        raise ValueError("payload['baseline_sweep'] must be a list")

    canonical_n = int(payload["canonical_n"])
    n_values = [int(n) for n in payload["n_values"]]

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        if "num_nodes" not in rec:
            raise KeyError("sweep record missing 'num_nodes'")

    metrics: dict[str, float | int] = {"version": VERSION}

    # --- Per-graph-size slices ---
    for n in n_values:
        recs = [r for r in sweep if int(r.get("num_nodes", -1)) == n]
        metrics[f"color_separation_n_{n}"] = _mean([_separation(r) for r in recs])
        metrics[f"edge_respect_n_{n}"] = _mean([_edge_respect(r) for r in recs])

    # --- Canonical slice (canonical_n) ---
    canon = [r for r in sweep if int(r.get("num_nodes", -1)) == canonical_n]
    metrics["color_separation_canonical"] = _mean([_separation(r) for r in canon])
    metrics["edge_respect_canonical"] = _mean([_edge_respect(r) for r in canon])
    metrics["invalid_edge_attention_canonical"] = _mean(
        [float(r.get("cross_edge_same_color", 0.0)) for r in canon]
    )

    # --- No-mechanism baseline, same canonical condition ---
    base_canon = [r for r in baseline_sweep if int(r.get("num_nodes", -1)) == canonical_n]
    metrics["linear_baseline_color_separation"] = _mean(
        [_separation(r) for r in base_canon]
    )
    metrics["lift_over_linear_baseline"] = (
        metrics["color_separation_canonical"]
        - metrics["linear_baseline_color_separation"]
    )

    # --- Overall summary across all graphs ---
    metrics["color_separation_overall"] = _mean([_separation(r) for r in sweep])
    metrics["num_graphs"] = int(len(sweep))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are mechanically degenerate, to skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    sep = metrics.get("color_separation_canonical")
    baseline = metrics.get("linear_baseline_color_separation")
    if isinstance(sep, (int, float)) and isinstance(baseline, (int, float)):
        # Failing to beat the structureless uniform baseline at the canonical
        # condition is a mechanical failure worth short-circuiting the jury.
        if sep <= baseline:
            return True

    return False
