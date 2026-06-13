"""
Benchmark for `attention_int_add`.

Consumes the payload returned by `task.evaluate` and produces a flat dict of
named scalar metrics. See README.md for the contract.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _carry_key(prefix: str, k: int) -> str:
    """Slice key name, e.g. ('exact_match', 3) -> 'exact_match_carries_3'."""
    return f"{prefix}_carries_{int(k)}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), max_digits (int), sum_digits (int),
        canonical_carries (int), carry_sweep (list[int]),
        sweep (list[record]), linear_baseline (list[record]).

    Each sweep record:          {carries, exact_match_rate, digit_accuracy, n}
    Each linear_baseline record:{carries, exact_match_rate, digit_accuracy, n}
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "canonical_carries", "carry_sweep",
                "sweep", "linear_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    carry_sweep = payload["carry_sweep"]
    if not isinstance(carry_sweep, (list, tuple)) or len(carry_sweep) == 0:
        raise ValueError("payload['carry_sweep'] must be a non-empty list")
    carry_sweep = [int(c) for c in carry_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(carry_sweep):
        raise ValueError("payload['sweep'] must be a list of same length as carry_sweep")

    linear_baseline = payload["linear_baseline"]
    if not isinstance(linear_baseline, (list, tuple)) or len(linear_baseline) != len(carry_sweep):
        raise ValueError(
            "payload['linear_baseline'] must be a list of same length as carry_sweep"
        )

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "carries" not in rec:
                raise KeyError(f"{what} record missing 'carries'")
            out[int(rec["carries"])] = rec
        return out

    sweep_by_k = _index(sweep, "sweep")
    base_by_k = _index(linear_baseline, "linear_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    for k in carry_sweep:
        srec = sweep_by_k.get(k, {})
        brec = base_by_k.get(k, {})

        em = float(srec.get("exact_match_rate", 0.0))
        dacc = float(srec.get("digit_accuracy", 0.0))
        b_em = float(brec.get("exact_match_rate", 0.0))

        metrics[_carry_key("exact_match", k)] = em
        metrics[_carry_key("digit_accuracy", k)] = dacc
        metrics[_carry_key("linear_baseline_exact_match", k)] = b_em

    # --- Canonical condition ---
    canonical_k = int(payload["canonical_carries"])
    metrics["exact_match_canonical"] = float(
        metrics.get(_carry_key("exact_match", canonical_k), 0.0)
    )
    base_canonical = float(
        metrics.get(_carry_key("linear_baseline_exact_match", canonical_k), 0.0)
    )
    metrics["lift_over_baseline_canonical"] = (
        metrics["exact_match_canonical"] - base_canonical
    )

    # --- Overall accuracy across all slices (unweighted mean) ---
    em_vals = [metrics[_carry_key("exact_match", k)] for k in carry_sweep]
    metrics["exact_match_mean"] = float(sum(em_vals) / len(em_vals))

    # --- Headline: carry_robustness ---
    # Exact-match retained at the hardest carry condition relative to the
    # no-carry condition. 1.0 => carries cost nothing; 0.0 => carries break it.
    em_easy = float(metrics.get(_carry_key("exact_match", carry_sweep[0]), 0.0))
    em_hard = float(metrics.get(_carry_key("exact_match", carry_sweep[-1]), 0.0))
    if em_easy > 1e-12:
        robustness = em_hard / em_easy
    else:
        robustness = 0.0
    metrics["carry_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that does not even beat the no-carry linear baseline at the
    # canonical (hardest) condition is mechanically degenerate.
    sharp = metrics.get("exact_match_canonical")
    baseline = metrics.get("linear_baseline_exact_match_carries_3")
    if isinstance(sharp, (int, float)) and isinstance(baseline, (int, float)):
        if sharp <= baseline:
            return True

    return False
