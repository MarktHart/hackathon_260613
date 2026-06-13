"""Benchmark for the attention_int_mul goal.

Pure Python. Deterministic. Side-effect free. Consumes the payload produced by
task.evaluate() and returns a flat dict of scalar metrics. See README.md.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _k_key(prefix: str, k: int) -> str:
    """Slice key name, e.g. ('routing_accuracy', 8) -> 'routing_accuracy_k_8'."""
    return f"{prefix}_k_{int(k)}"


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the task.evaluate() payload.

    Expected payload keys:
        version (int == 1), d (int), n_positions (int), canonical_k (int),
        k_sweep (list[int]), sweep (list[record]), linear_baseline (list[record]).

    Each sweep record:           {k, routing_accuracy, attended_mass, n_trials}.
    Each linear_baseline record: {k, routing_accuracy, n_trials}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "d", "n_positions", "canonical_k",
                "k_sweep", "sweep", "linear_baseline"]
    for key in required:
        if key not in payload:
            raise KeyError(f"Missing required payload key: {key!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    k_sweep = payload["k_sweep"]
    if not isinstance(k_sweep, (list, tuple)) or len(k_sweep) == 0:
        raise ValueError("payload['k_sweep'] must be a non-empty list")
    k_sweep = [int(k) for k in k_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(k_sweep):
        raise ValueError("payload['sweep'] must be a list of same length as k_sweep")

    linear_baseline = payload["linear_baseline"]
    if not isinstance(linear_baseline, (list, tuple)) or len(linear_baseline) != len(k_sweep):
        raise ValueError(
            "payload['linear_baseline'] must be a list of same length as k_sweep"
        )

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "k" not in rec:
                raise KeyError(f"{what} record missing 'k'")
            out[int(rec["k"])] = rec
        return out

    sweep_by_k = _index(sweep, "sweep")
    base_by_k = _index(linear_baseline, "linear_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    acc_vals = []
    base_vals = []

    for k in k_sweep:
        srec = sweep_by_k.get(k, {})
        brec = base_by_k.get(k, {})

        acc = float(srec.get("routing_accuracy", 0.0))
        mass = float(srec.get("attended_mass", 0.0))
        base_acc = float(brec.get("routing_accuracy", 0.0))

        metrics[_k_key("routing_accuracy", k)] = acc
        metrics[_k_key("attended_mass", k)] = mass
        metrics[_k_key("linear_baseline_accuracy", k)] = base_acc

        acc_vals.append(acc)
        base_vals.append(base_acc)

    # --- Canonical slice ---
    canonical_k = int(payload["canonical_k"])
    metrics["routing_accuracy_canonical"] = float(
        metrics.get(_k_key("routing_accuracy", canonical_k), 0.0)
    )
    metrics["attended_mass_canonical"] = float(
        metrics.get(_k_key("attended_mass", canonical_k), 0.0)
    )
    metrics["linear_baseline_accuracy_canonical"] = float(
        metrics.get(_k_key("linear_baseline_accuracy", canonical_k), 0.0)
    )
    metrics["lift_over_baseline_canonical"] = (
        metrics["routing_accuracy_canonical"]
        - metrics["linear_baseline_accuracy_canonical"]
    )

    # --- Aggregates ---
    metrics["mean_routing_accuracy"] = (
        float(sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0
    )
    metrics["mean_linear_baseline_accuracy"] = (
        float(sum(base_vals) / len(base_vals)) if base_vals else 0.0
    )

    # --- Robustness: accuracy at hardest K / accuracy at easiest K ---
    acc_easy = float(metrics.get(_k_key("routing_accuracy", k_sweep[0]), 0.0))
    acc_hard = float(metrics.get(_k_key("routing_accuracy", k_sweep[-1]), 0.0))
    if acc_easy > 1e-12:
        robustness = acc_hard / acc_easy
    else:
        robustness = 0.0
    metrics["routing_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Pipeline hook: True if metrics are clearly degenerate, to skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # The mechanically-degenerate case: the attempt does not even beat the
    # no-mechanism additive baseline on average across the sweep.
    mean_acc = metrics.get("mean_routing_accuracy")
    mean_base = metrics.get("mean_linear_baseline_accuracy")
    if isinstance(mean_acc, (int, float)) and isinstance(mean_base, (int, float)):
        if mean_acc <= mean_base:
            return True

    return False
