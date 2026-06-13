"""
Benchmark for attention_tsp.

Consumes the payload returned by task.evaluate() and produces a flat dict of
scalar metrics. Pure Python, deterministic, side-effect free.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _n_key(prefix: str, n: int) -> str:
    """Slice key name, e.g. ('nn_accuracy', 10) -> 'nn_accuracy_n_10'."""
    return f"{prefix}_n_{int(n)}"


def _index_by_n(records, what: str) -> dict:
    out = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise ValueError(f"Each {what} record must be a dict, got {type(rec).__name__}")
        if "n" not in rec:
            raise KeyError(f"{what} record missing 'n'")
        out[int(rec["n"])] = rec
    return out


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the task.evaluate() payload.

    Expected payload keys:
        version (int == 1), model_name (str), canonical_n (int),
        n_cities_sweep (list[int]), sweep (list[record]),
        random_baseline (list[record]).

    Each sweep record:           {n, nn_accuracy, tour_length_ratio, n_instances}
    Each random_baseline record: {n, nn_accuracy, tour_length_ratio, n_instances}
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "canonical_n", "n_cities_sweep", "sweep", "random_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    n_sweep = payload["n_cities_sweep"]
    if not isinstance(n_sweep, (list, tuple)) or len(n_sweep) == 0:
        raise ValueError("payload['n_cities_sweep'] must be a non-empty list")
    n_sweep = [int(n) for n in n_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(n_sweep):
        raise ValueError("payload['sweep'] must be a list of same length as n_cities_sweep")

    baseline = payload["random_baseline"]
    if not isinstance(baseline, (list, tuple)) or len(baseline) != len(n_sweep):
        raise ValueError(
            "payload['random_baseline'] must be a list of same length as n_cities_sweep"
        )

    sweep_by_n = _index_by_n(sweep, "sweep")
    base_by_n = _index_by_n(baseline, "random_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    acc_vals = []
    ratio_vals = []
    base_acc_vals = []

    for n in n_sweep:
        srec = sweep_by_n.get(n, {})
        brec = base_by_n.get(n, {})

        acc = float(srec.get("nn_accuracy", 0.0))
        ratio = float(srec.get("tour_length_ratio", 0.0))
        b_acc = float(brec.get("nn_accuracy", 0.0))
        b_ratio = float(brec.get("tour_length_ratio", 0.0))

        metrics[_n_key("nn_accuracy", n)] = acc
        metrics[_n_key("tour_length_ratio", n)] = ratio
        metrics[_n_key("random_baseline_nn_accuracy", n)] = b_acc
        metrics[_n_key("random_baseline_tour_length_ratio", n)] = b_ratio

        acc_vals.append(acc)
        ratio_vals.append(ratio)
        base_acc_vals.append(b_acc)

    # --- Canonical slice ---
    canonical_n = int(payload["canonical_n"])
    metrics["nn_accuracy_canonical"] = float(metrics.get(_n_key("nn_accuracy", canonical_n), 0.0))
    metrics["tour_length_ratio_canonical"] = float(
        metrics.get(_n_key("tour_length_ratio", canonical_n), 0.0)
    )
    base_canonical = float(metrics.get(_n_key("random_baseline_nn_accuracy", canonical_n), 0.0))
    metrics["lift_over_baseline_canonical"] = (
        metrics["nn_accuracy_canonical"] - base_canonical
    )

    # --- Aggregate over the whole sweep ---
    metrics["nn_accuracy_mean"] = float(sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0
    metrics["tour_length_ratio_mean"] = (
        float(sum(ratio_vals) / len(ratio_vals)) if ratio_vals else 0.0
    )

    # --- Headline: size_robustness ---
    # Step-wise NN accuracy retained at the largest problem size relative to the
    # smallest. 1.0 = the mechanism scales perfectly; -> 0 = it falls apart as
    # the city count grows. Clipped to [0, 1].
    acc_small = float(metrics.get(_n_key("nn_accuracy", n_sweep[0]), 0.0))
    acc_large = float(metrics.get(_n_key("nn_accuracy", n_sweep[-1]), 0.0))
    if acc_small > 1e-12:
        robustness = acc_large / acc_small
    else:
        robustness = 0.0
    metrics["size_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are clearly degenerate, so the pipeline can skip the jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # A mechanism that does not even beat random city selection at the canonical
    # condition is mechanically degenerate — no model judgement needed.
    acc = metrics.get("nn_accuracy_canonical")
    base = metrics.get("random_baseline_nn_accuracy_n_10")
    if isinstance(acc, (int, float)) and isinstance(base, (int, float)):
        if acc <= base:
            return True

    return False
