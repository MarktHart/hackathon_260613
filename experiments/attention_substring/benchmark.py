"""Benchmark for the attention_substring goal.

Pure Python. Deterministic. Side-effect free. No imports from any attempt dir.

The payload (produced by task.evaluate) is a `sweep` of per-sequence records,
one per generated sequence, plus a self-describing `config`. Each record stores
the *best head*'s behaviour at the target position: whether it attended to the
correct source position (`correct_top1`), how strong that attention was relative
to the best distractor (`attn_to_source` / `max_attn_elsewhere`), and an optional
next-token prediction.

We score how reliably the model implements substring matching (induction-style
copying), broken down by pattern length and inter-occurrence distance, against a
uniform-attention chance baseline.
"""

from __future__ import annotations

import math
from typing import Any

VERSION = 1

# Canonical axes — must match task.py.
PATTERN_LENGTHS: list[int] = [2, 3, 4]
DISTANCES: list[int] = [8, 16, 32]
_ATTN_RATIO_CLIP: float = 100.0

_REQUIRED_RECORD_KEYS = (
    "pattern_length",
    "distance",
    "attn_to_source",
    "max_attn_elsewhere",
    "correct_top1",
    "target_token",
    "predicted_token",
)


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v: Any) -> bool:
    return _num(v) and not (math.isnan(float(v)) or math.isinf(float(v)))


def _mean(xs: list[float]) -> float:
    """Mean with explicit empty handling (0.0, never a ZeroDivisionError)."""
    return float(sum(xs) / len(xs)) if xs else 0.0


def _validate(payload: dict[str, Any]) -> list[dict]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(
            f"payload version {payload.get('version')!r} != benchmark VERSION {VERSION}"
        )
    sweep = payload.get("sweep")
    if not isinstance(sweep, list):
        raise KeyError("payload missing 'sweep' (must be a list)")
    if len(sweep) == 0:
        raise ValueError("payload['sweep'] is empty; nothing to score")
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict, got {type(rec).__name__}")
        for k in _REQUIRED_RECORD_KEYS:
            if k not in rec:
                raise KeyError(f"sweep[{i}] missing key {k!r}")
        for k in ("attn_to_source", "max_attn_elsewhere"):
            if not _finite(rec[k]):
                raise ValueError(f"sweep[{i}][{k!r}] must be a finite number, got {rec[k]!r}")
    return sweep


def _attn_ratio(rec: dict) -> float:
    """attn_to_source / max_attn_elsewhere, clipped to [0, _ATTN_RATIO_CLIP].

    The denominator is a max over the (seq_len - 2) non-source, non-target
    positions; it can legitimately be ~0 when the head ignores everything else.
    We clip rather than divide-by-zero so the metric stays bounded."""
    src = float(rec["attn_to_source"])
    other = float(rec["max_attn_elsewhere"])
    if other <= 1e-12:
        return _ATTN_RATIO_CLIP if src > 0.0 else 0.0
    return float(min(_ATTN_RATIO_CLIP, max(0.0, src / other)))


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    sweep = _validate(payload)

    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    seq_len = config.get("seq_len", 64)
    if not (_num(seq_len) and seq_len > 1):
        seq_len = 64

    metrics: dict[str, float | int] = {"version": VERSION}

    # ---- headline: overall detection rate ----
    all_correct = [1.0 if rec.get("correct_top1") else 0.0 for rec in sweep]
    detection_canonical = _mean(all_correct)
    metrics["substring_detection_canonical"] = detection_canonical

    # ---- per (pattern_length, distance) slices ----
    for L in PATTERN_LENGTHS:
        for D in DISTANCES:
            cell = [
                1.0 if rec.get("correct_top1") else 0.0
                for rec in sweep
                if rec.get("pattern_length") == L and rec.get("distance") == D
            ]
            metrics[f"substring_detection_plen_{L}_dist_{D}"] = _mean(cell)

    # ---- marginal slices: by pattern length ----
    for L in PATTERN_LENGTHS:
        cell = [1.0 if rec.get("correct_top1") else 0.0 for rec in sweep if rec.get("pattern_length") == L]
        metrics[f"substring_detection_plen_{L}"] = _mean(cell)

    # ---- marginal slices: by distance ----
    for D in DISTANCES:
        cell = [1.0 if rec.get("correct_top1") else 0.0 for rec in sweep if rec.get("distance") == D]
        metrics[f"substring_detection_dist_{D}"] = _mean(cell)

    # ---- attention strength ratio ----
    metrics["attn_ratio_canonical"] = _mean([_attn_ratio(rec) for rec in sweep])

    # ---- optional next-token accuracy (only when logits were provided) ----
    pred = [
        1.0 if int(rec["predicted_token"]) == int(rec["target_token"]) else 0.0
        for rec in sweep
        if _num(rec.get("predicted_token")) and int(rec["predicted_token"]) >= 0
    ]
    if pred:
        metrics["token_prediction_accuracy"] = _mean(pred)

    # ---- chance baseline + lift ----
    random_baseline = 1.0 / float(seq_len - 1)
    metrics["random_baseline_detection"] = float(random_baseline)
    metrics["lift_over_random"] = float(detection_canonical - random_baseline)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Mechanical short-circuit before the (expensive) jury. Must never return
    True for a borderline-but-real result — only clearly degenerate ones."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    det = metrics.get("substring_detection_canonical")
    base = metrics.get("random_baseline_detection")
    if _num(det) and _num(base):
        if det <= max(base, 0.0) * 1.5:
            return True
    acc = metrics.get("token_prediction_accuracy")
    if _num(acc) and acc < 0.01:
        return True
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
