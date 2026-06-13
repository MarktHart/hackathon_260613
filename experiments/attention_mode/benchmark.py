"""
Scoring for the `attention_mode` goal.

Consumes the payload produced by `task.evaluate` (a noise sweep of per-head
mode predictions) and returns a flat dict of named scalar metrics.

Pure Python. Deterministic. No I/O, no imports from any attempt directory.
"""
from __future__ import annotations

import math

VERSION = 1

# Pipeline hint: attempts run on the GPU. task.py / benchmark.py stay CPU-only.
GPU_REQUIREMENT = 1


def _fmt(x: float) -> str:
    """Format a float for a metric key: 0.0 -> '0p0', 0.5 -> '0p5'."""
    s = "%g" % x
    if "." not in s and "e" not in s:
        s += ".0"
    return s.replace("-", "neg").replace(".", "p")


def _argmax_mode(pred_probs: dict) -> str:
    """Mode with the highest predicted probability (ties broken by key order)."""
    return max(pred_probs, key=lambda m: (pred_probs[m], m))


def score(payload: dict) -> dict[str, float | int]:
    # ---- Contract validation -------------------------------------------------
    required = {"version", "L", "seed", "modes", "noise_levels", "sweep"}
    missing = required - set(payload.keys())
    if missing:
        raise KeyError(f"Payload missing keys: {sorted(missing)}")
    if payload["version"] != VERSION:
        raise ValueError(
            f"Payload version {payload['version']} != benchmark VERSION {VERSION}"
        )

    modes = list(payload["modes"])
    if not modes:
        raise ValueError("modes must be non-empty")
    mode_set = set(modes)
    n_modes = len(modes)

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("sweep must be a non-empty list")

    noise_levels = list(payload["noise_levels"])
    if not noise_levels:
        raise ValueError("noise_levels must be non-empty")
    canonical_noise = float(payload.get("canonical_noise", min(noise_levels)))

    for i, rec in enumerate(sweep):
        for k in ("noise", "true_mode", "pred_probs"):
            if k not in rec:
                raise KeyError(f"Record {i} missing field '{k}'")
        if rec["true_mode"] not in mode_set:
            raise ValueError(f"Record {i}: true_mode '{rec['true_mode']}' not in modes")
        if set(rec["pred_probs"].keys()) != mode_set:
            raise ValueError(f"Record {i}: pred_probs keys do not match modes")

    # ---- Group records by noise slice ---------------------------------------
    by_noise: dict[float, list[dict]] = {}
    for rec in sweep:
        by_noise.setdefault(float(rec["noise"]), []).append(rec)

    def _accuracy(records: list[dict]) -> float:
        if not records:
            return 0.0
        hits = sum(1 for r in records if _argmax_mode(r["pred_probs"]) == r["true_mode"])
        return hits / len(records)

    def _cross_entropy(records: list[dict]) -> float:
        # Mean negative log-prob of the true mode (clamped to avoid -inf).
        if not records:
            return 0.0
        total = 0.0
        for r in records:
            p = r["pred_probs"][r["true_mode"]]
            total += -math.log(max(p, 1e-12))
        return total / len(records)

    def _macro_f1(records: list[dict]) -> float:
        if not records:
            return 0.0
        f1s = []
        for m in modes:
            tp = sum(1 for r in records
                     if r["true_mode"] == m and _argmax_mode(r["pred_probs"]) == m)
            fp = sum(1 for r in records
                     if r["true_mode"] != m and _argmax_mode(r["pred_probs"]) == m)
            fn = sum(1 for r in records
                     if r["true_mode"] == m and _argmax_mode(r["pred_probs"]) != m)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec_ / (prec + rec_) if (prec + rec_) > 0 else 0.0
            f1s.append(f1)
        return sum(f1s) / len(f1s)

    # ---- Metrics -------------------------------------------------------------
    metrics: dict[str, float | int] = {"version": VERSION}

    # Canonical slice (clean patterns).
    canonical_records = by_noise.get(canonical_noise, [])
    metrics["accuracy_canonical"] = _accuracy(canonical_records)
    metrics["macro_f1_canonical"] = _macro_f1(canonical_records)
    metrics["cross_entropy_canonical"] = _cross_entropy(canonical_records)

    # Per-mode accuracy at the canonical slice.
    for m in modes:
        recs_m = [r for r in canonical_records if r["true_mode"] == m]
        metrics[f"accuracy_mode_{m}"] = _accuracy(recs_m)

    # Per-slice accuracy across the noise sweep.
    slice_accs: dict[float, float] = {}
    for noise in sorted(by_noise.keys()):
        acc = _accuracy(by_noise[noise])
        slice_accs[noise] = acc
        metrics[f"accuracy_noise_{_fmt(noise)}"] = acc

    # Headline robustness: accuracy retained at the hardest noise relative to
    # the canonical (clean) accuracy, clamped to [0, 1]. 1.0 means no
    # degradation; 0.0 means the mechanism collapses under corruption.
    hardest = max(by_noise.keys())
    base_acc = metrics["accuracy_canonical"]
    if base_acc <= 0.0:
        metrics["mode_robustness"] = 0.0
    else:
        metrics["mode_robustness"] = max(0.0, min(1.0, slice_accs[hardest] / base_acc))

    # Reference baseline: random guessing under identical conditions.
    metrics["linear_baseline_accuracy_canonical"] = 1.0 / n_modes
    metrics["lift_over_baseline_accuracy"] = (
        metrics["accuracy_canonical"] - metrics["linear_baseline_accuracy_canonical"]
    )

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Mechanical degeneracy check; lets the pipeline skip the jury on clearly
    failed attempts. Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    acc = metrics.get("accuracy_canonical")
    baseline = metrics.get("linear_baseline_accuracy_canonical")
    # A real mechanism on clean patterns should comfortably beat random.
    if isinstance(acc, (int, float)) and isinstance(baseline, (int, float)):
        if acc <= baseline * 1.5:
            return True
    return False
