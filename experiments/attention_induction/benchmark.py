"""Benchmark for the attention_induction goal.

Pure Python, deterministic, side-effect free. Consumes the payload returned by
task.evaluate() and emits a flat dict of scalar metrics.
"""

import math

VERSION = 1

GPU_REQUIREMENT = 1  # attempts run a real model on the GPU; minimum is 1


def _dist_key(prefix: str, distance: int) -> str:
    return f"{prefix}_dist_{int(distance)}"


def score(payload: dict) -> dict[str, float | int]:
    """Compute metrics from a task.evaluate() payload.

    Expected payload keys:
        version (int == 1), model_name, vocab_size, seq_len,
        canonical_distance (int),
        sweep: list of {distance, num_targets, accuracy, ce_loss,
                        uniform_baseline_accuracy, uniform_baseline_ce_loss},
        aggregate: {accuracy, ce_loss, num_targets,
                    uniform_baseline_accuracy, uniform_baseline_ce_loss}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_distance", "sweep", "aggregate"):
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

    aggregate = payload["aggregate"]
    if not isinstance(aggregate, dict):
        raise ValueError("payload['aggregate'] must be a dict")
    for k in ("accuracy", "ce_loss", "num_targets",
              "uniform_baseline_accuracy", "uniform_baseline_ce_loss"):
        if k not in aggregate:
            raise KeyError(f"payload['aggregate'] missing {k!r}")

    canonical_distance = int(payload["canonical_distance"])

    metrics: dict[str, float | int] = {"version": VERSION}

    # ---- Headline + aggregate ----
    overall_acc = float(aggregate["accuracy"])
    overall_ce = float(aggregate["ce_loss"])
    baseline_acc = float(aggregate["uniform_baseline_accuracy"])
    baseline_ce = float(aggregate["uniform_baseline_ce_loss"])

    metrics["induction_accuracy"] = overall_acc                  # headline, bigger=better
    metrics["induction_ce_loss"] = overall_ce                    # smaller=better
    metrics["uniform_baseline_accuracy"] = baseline_acc          # reference
    metrics["uniform_baseline_ce_loss"] = baseline_ce            # reference
    metrics["lift_over_uniform"] = overall_acc - baseline_acc    # bigger=better
    metrics["num_targets"] = int(aggregate["num_targets"])

    # ---- Per-slice (per distance) ----
    acc_by_dist: dict[int, float] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        for k in ("distance", "accuracy", "ce_loss"):
            if k not in rec:
                raise KeyError(f"sweep record missing {k!r}")
        d = int(rec["distance"])
        acc = float(rec["accuracy"])
        loss = float(rec["ce_loss"])
        acc_by_dist[d] = acc
        metrics[_dist_key("induction_accuracy", d)] = acc
        metrics[_dist_key("induction_ce_loss", d)] = loss
        metrics[_dist_key("num_targets", d)] = int(rec.get("num_targets", 0))

    # Canonical slice value.
    metrics["induction_accuracy_canonical"] = acc_by_dist.get(canonical_distance, 0.0)

    # ---- Distance robustness: accuracy at the largest vs smallest distance ----
    distances = sorted(acc_by_dist.keys())
    if distances:
        near = acc_by_dist[distances[0]]
        far = acc_by_dist[distances[-1]]
        if near > 0:
            metrics["distance_robustness"] = max(0.0, min(1.0, far / near))
        else:
            metrics["distance_robustness"] = 0.0
    else:
        metrics["distance_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Short-circuit the jury for clearly degenerate attempts."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    acc = metrics.get("induction_accuracy")
    baseline = metrics.get("uniform_baseline_accuracy")
    if isinstance(acc, (int, float)) and isinstance(baseline, (int, float)):
        # Real induction must clear the uniform baseline by a clear margin.
        if acc <= baseline * 1.5:
            return True

    lift = metrics.get("lift_over_uniform")
    if isinstance(lift, (int, float)) and lift <= 0:
        return True

    return False
