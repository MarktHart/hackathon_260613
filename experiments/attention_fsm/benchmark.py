"""Benchmark for the attention_fsm goal: DFA state-tracking metrics.

Pure Python, deterministic, side-effect free. Consumes the payload returned by
task.evaluate; never imports from any attempt directory.
"""

import math

VERSION = 1

# Positions (post-burn-in) at which we report per-slice accuracy.
_SLICE_POSITIONS = (16, 24, 32, 40, 48, 56, 63)


def _require(payload: dict, key: str):
    if key not in payload:
        raise KeyError(f"payload missing required key {key!r}")
    return payload[key]


def score(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    seq_len = int(_require(payload, "seq_len"))
    burnin = int(_require(payload, "burnin"))
    overall = float(_require(payload, "overall_accuracy"))
    chance = float(_require(payload, "random_baseline_accuracy"))
    per_pos = _require(payload, "per_position_accuracy")
    per_state_recall = _require(payload, "per_state_recall")

    if not isinstance(per_pos, list) or len(per_pos) != seq_len:
        raise ValueError(
            f"per_position_accuracy must be a list of length seq_len={seq_len}, "
            f"got length {len(per_pos) if isinstance(per_pos, list) else 'n/a'}"
        )
    if not isinstance(per_state_recall, list) or len(per_state_recall) == 0:
        raise ValueError("per_state_recall must be a non-empty list")
    if not 0 <= burnin < seq_len:
        raise ValueError(f"burnin {burnin} out of range for seq_len {seq_len}")

    metrics: dict = {"version": VERSION}

    metrics["state_tracking_accuracy_canonical"] = overall
    metrics["random_baseline_accuracy"] = chance
    metrics["lift_over_random"] = overall - chance

    # Headline: chance-normalised accuracy, clamped to [0, 1].
    denom = 1.0 - chance
    if denom <= 0.0:
        robustness = 0.0
    else:
        robustness = (overall - chance) / denom
    metrics["state_tracking_robustness"] = float(min(1.0, max(0.0, robustness)))

    # Per-slice accuracy at sampled positions.
    for p in _SLICE_POSITIONS:
        if 0 <= p < seq_len:
            metrics[f"acc_pos_{p}"] = float(per_pos[p])

    # Per-state recall + weakest state (collapse detector).
    for s, r in enumerate(per_state_recall):
        metrics[f"state_recall_{s}"] = float(r)
    metrics["min_state_recall"] = float(min(per_state_recall))

    # Depth stability: late quarter vs first post-burn-in quarter.
    post = per_pos[burnin:]
    if len(post) >= 4:
        q = max(1, len(post) // 4)
        early = sum(post[:q]) / q
        late = sum(post[-q:]) / q
        metrics["late_minus_early_accuracy"] = float(late - early)
    else:
        metrics["late_minus_early_accuracy"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    # NaN / inf anywhere is a hard failure.
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # No better than chance => the mechanism does nothing.
    lift = metrics.get("lift_over_random")
    if isinstance(lift, int | float) and lift <= 0.0:
        return True
    # Collapse onto a subset of states (one state never recalled).
    min_recall = metrics.get("min_state_recall")
    if isinstance(min_recall, int | float) and min_recall <= 0.0:
        return True
    return False
