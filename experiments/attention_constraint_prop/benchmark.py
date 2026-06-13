"""Benchmark for the attention constraint-propagation goal.

Consumes the payload produced by ``task.evaluate`` and returns a flat dict of
scalar metrics.  Pure Python; deterministic; no I/O.
"""

import math

VERSION = 1

# Pipeline-only hook: GPU slots the attempt subprocess needs (min 1).
GPU_REQUIREMENT = 1


def _dist_key(prefix: str, distance: int) -> str:
    """Per-slice key, e.g. ('mean_alignment', 4) -> 'mean_alignment_dist_4'."""
    return f"{prefix}_dist_{int(distance)}"


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from a ``task.evaluate`` payload.

    Expected payload keys:
        version (int == 1),
        config (dict: seq_len, num_sequences, constraint_types,
                canonical_distance, seed),
        model_info (dict: n_layers, n_heads),
        sweep (list of per-distance records, each:
               {distance, heads, mean_alignment, max_alignment, best_head}).
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "config", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    config = payload["config"]
    if not isinstance(config, dict):
        raise ValueError("payload['config'] must be a dict")

    seq_len = int(config.get("seq_len", 0))
    if seq_len <= 0:
        raise ValueError(f"config['seq_len'] must be positive, got {seq_len!r}")
    canonical_distance = int(config.get("canonical_distance", 0))

    baseline = 1.0 / float(seq_len)          # uniform-attention alignment

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")

    metrics: dict[str, float | int] = {"version": VERSION}
    metrics["canonical_distance"] = canonical_distance
    metrics["baseline_alignment_canonical"] = baseline

    canonical_record = None
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        if "distance" not in rec:
            raise KeyError("sweep record missing 'distance'")
        d = int(rec["distance"])
        mean_a = float(rec.get("mean_alignment", 0.0))
        max_a = float(rec.get("max_alignment", 0.0))

        metrics[_dist_key("mean_alignment", d)] = mean_a
        metrics[_dist_key("max_alignment", d)] = max_a
        metrics[_dist_key("random_baseline_alignment", d)] = baseline

        if d == canonical_distance:
            canonical_record = rec

    # --- Canonical / headline ---
    if canonical_record is not None:
        max_canonical = float(canonical_record.get("max_alignment", 0.0))
        best = canonical_record.get("best_head", {}) or {}
        best_layer = int(best.get("layer", -1))
        best_head = int(best.get("head", -1))
    else:
        # Canonical distance absent from the sweep: degrade gracefully.
        max_canonical = 0.0
        best_layer = -1
        best_head = -1

    metrics["max_head_alignment_canonical"] = max_canonical
    metrics["best_head_layer_canonical"] = best_layer
    metrics["best_head_head_canonical"] = best_head

    if baseline > 0.0:
        fidelity = max_canonical / baseline
    else:
        fidelity = 0.0
    metrics["constraint_propagation_fidelity"] = float(fidelity)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are mechanically degenerate, so the jury can be skipped.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # A real constraint-propagating head must beat uniform attention, whose
    # fidelity is exactly 1.0 by construction.  Anything at-or-below random
    # carries no mechanism worth jurying.
    fidelity = metrics.get("constraint_propagation_fidelity")
    if isinstance(fidelity, (int, float)) and fidelity <= 1.0:
        return True

    return False
