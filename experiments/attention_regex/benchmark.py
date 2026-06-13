"""Benchmark for the attention_regex goal.

Consumes the payload from task.evaluate() and returns flat scalar metrics.
Pure Python, deterministic, side-effect free. No imports from any attempt
directory.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _len_key(prefix: str, L: int) -> str:
    """Slice key, e.g. ('match_sharpness', 3) -> 'match_sharpness_len_3'."""
    return f"{prefix}_len_{int(L)}"


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the task.evaluate() payload.

    Expected payload keys:
        version (int == 1), d (int), vocab_size (int), n_positions (int),
        canonical_length (int), length_sweep (list[int]),
        sweep (list[record]), linear_baseline (list[record]).

    Each sweep record: {length, match_sharpness, false_positive_rate,
                        false_negative_rate, n_seeds}.
    Each linear_baseline record: {length, match_sharpness, n_seeds}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "d", "canonical_length", "length_sweep",
                "sweep", "linear_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    d = payload["d"]
    if not isinstance(d, (int, float)) or d <= 0:
        raise ValueError(f"payload['d'] must be a positive number, got {d!r}")

    length_sweep = payload["length_sweep"]
    if not isinstance(length_sweep, (list, tuple)) or len(length_sweep) == 0:
        raise ValueError("payload['length_sweep'] must be a non-empty list")
    length_sweep = [int(L) for L in length_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(length_sweep):
        raise ValueError("payload['sweep'] must be a list as long as length_sweep")

    linear_baseline = payload["linear_baseline"]
    if not isinstance(linear_baseline, (list, tuple)) or len(linear_baseline) != len(length_sweep):
        raise ValueError(
            "payload['linear_baseline'] must be a list as long as length_sweep"
        )

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "length" not in rec:
                raise KeyError(f"{what} record missing 'length'")
            out[int(rec["length"])] = rec
        return out

    sweep_by_len = _index(sweep, "sweep")
    base_by_len = _index(linear_baseline, "linear_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    for L in length_sweep:
        srec = sweep_by_len.get(L, {})
        brec = base_by_len.get(L, {})

        m_sharp = float(srec.get("match_sharpness", 0.0))
        fpr = float(srec.get("false_positive_rate", 1.0))
        fnr = float(srec.get("false_negative_rate", 1.0))
        b_sharp = float(brec.get("match_sharpness", 0.0))

        metrics[_len_key("match_sharpness", L)] = m_sharp
        metrics[_len_key("false_positive_rate", L)] = fpr
        metrics[_len_key("false_negative_rate", L)] = fnr
        metrics[_len_key("linear_baseline_sharpness", L)] = b_sharp

    # --- Canonical condition ---
    canonical_L = int(payload["canonical_length"])
    metrics["match_sharpness_canonical"] = float(
        metrics.get(_len_key("match_sharpness", canonical_L), 0.0)
    )
    baseline_canonical = float(
        metrics.get(_len_key("linear_baseline_sharpness", canonical_L), 0.0)
    )
    metrics["lift_over_baseline_canonical"] = (
        metrics["match_sharpness_canonical"] - baseline_canonical
    )

    # --- Headline: length_robustness ---
    # Sharpness retained at the longest pattern vs the shortest one, in [0, 1].
    sharp_short = float(metrics.get(_len_key("match_sharpness", length_sweep[0]), 0.0))
    sharp_long = float(metrics.get(_len_key("match_sharpness", length_sweep[-1]), 0.0))
    if sharp_short > 1e-12:
        robustness = sharp_long / sharp_short
    else:
        robustness = 0.0
    metrics["length_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are clearly degenerate, to skip the (expensive) jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # match_sharpness is clipped to [0, 1]; a mechanism that cannot even beat
    # the no-composition linear baseline at the canonical length is degenerate.
    sharp = metrics.get("match_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_len_3")
    if isinstance(sharp, (int, float)) and isinstance(baseline, (int, float)):
        if baseline > 0 and sharp <= baseline:
            return True
        if baseline <= 0 and sharp <= 0:
            return True

    return False
