"""
attention_histogram — benchmark: payload -> flat scalar metrics.

Pure Python. No imports from any attempt directory. Deterministic, side-effect
free, defensive on its inputs.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _sim_key(prefix: str, sim: float) -> str:
    """Slice key, e.g. ('attention_sharpness', 0.2) -> 'attention_sharpness_sim_0p2'."""
    return f"{prefix}_sim_{sim:.1f}".replace(".", "p")


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), d (int), n_positions (int),
        canonical_similarity (float), key_sim_sweep (list[float]),
        sweep (list[record]), linear_baseline (list[record]).

    sweep record: {similarity, attention_sharpness, attention_entropy,
                   target_hit_rate, n_seeds}.
    linear_baseline record: {similarity, attention_sharpness,
                             target_hit_rate, n_seeds}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "d", "n_positions", "canonical_similarity",
                "key_sim_sweep", "sweep", "linear_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(
            f"Unsupported payload version: {version}. Expected {VERSION}.")

    n_positions = payload["n_positions"]
    if not isinstance(n_positions, (int, float)) or n_positions <= 0:
        raise ValueError(
            f"payload['n_positions'] must be positive, got {n_positions!r}")

    sim_sweep = payload["key_sim_sweep"]
    if not isinstance(sim_sweep, (list, tuple)) or len(sim_sweep) == 0:
        raise ValueError("payload['key_sim_sweep'] must be a non-empty list")
    sim_sweep = [float(c) for c in sim_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(sim_sweep):
        raise ValueError(
            "payload['sweep'] must be a list of same length as key_sim_sweep")

    linear_baseline = payload["linear_baseline"]
    if not isinstance(linear_baseline, (list, tuple)) \
            or len(linear_baseline) != len(sim_sweep):
        raise ValueError(
            "payload['linear_baseline'] must match key_sim_sweep length")

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "similarity" not in rec:
                raise KeyError(f"{what} record missing 'similarity'")
            out[round(float(rec["similarity"]), 6)] = rec
        return out

    sweep_by = _index(sweep, "sweep")
    base_by = _index(linear_baseline, "linear_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    for sim in sim_sweep:
        key = round(sim, 6)
        srec = sweep_by.get(key, {})
        brec = base_by.get(key, {})

        metrics[_sim_key("attention_sharpness", sim)] = float(
            srec.get("attention_sharpness", 0.0))
        metrics[_sim_key("attention_entropy", sim)] = float(
            srec.get("attention_entropy", 0.0))
        metrics[_sim_key("target_hit_rate", sim)] = float(
            srec.get("target_hit_rate", 0.0))
        metrics[_sim_key("linear_baseline_sharpness", sim)] = float(
            brec.get("attention_sharpness", 0.0))
        metrics[_sim_key("linear_baseline_hit_rate", sim)] = float(
            brec.get("target_hit_rate", 0.0))

    # --- Canonical condition ---
    canonical_sim = float(payload["canonical_similarity"])
    sharp_canonical = float(
        metrics.get(_sim_key("attention_sharpness", canonical_sim), 0.0))
    hit_canonical = float(
        metrics.get(_sim_key("target_hit_rate", canonical_sim), 0.0))
    base_sharp_canonical = float(
        metrics.get(_sim_key("linear_baseline_sharpness", canonical_sim), 0.0))

    metrics["attention_sharpness_canonical"] = sharp_canonical
    metrics["target_hit_rate_canonical"] = hit_canonical
    metrics["lift_over_baseline_canonical"] = sharp_canonical - base_sharp_canonical

    # --- Headline: histogram_robustness ---
    # Sharpness retained at the hardest condition (max key similarity, last)
    # relative to the easiest (distinct keys, first). In [0, 1].
    sharp_easy = float(
        metrics.get(_sim_key("attention_sharpness", sim_sweep[0]), 0.0))
    sharp_hard = float(
        metrics.get(_sim_key("attention_sharpness", sim_sweep[-1]), 0.0))
    if sharp_easy > 1e-12:
        robustness = sharp_hard / sharp_easy
    else:
        robustness = 0.0
    metrics["histogram_robustness"] = float(max(0.0, min(1.0, robustness)))

    # Mean hit-rate across the whole sweep — overall targeting accuracy.
    hit_vals = [float(metrics[_sim_key("target_hit_rate", s)]) for s in sim_sweep]
    metrics["mean_target_hit_rate"] = float(sum(hit_vals) / len(hit_vals))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True only for mechanically-degenerate results, to skip the (costly) jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # A real attempt must, at minimum, produce a non-uniform attention
    # histogram and beat chance targeting at the easy canonical condition.
    sharp = metrics.get("attention_sharpness_canonical")
    if isinstance(sharp, (int, float)) and sharp <= 1e-6:
        return True

    hit = metrics.get("target_hit_rate_canonical")
    # Chance at canonical is 1/n_positions; the canonical sweep uses 16 keys.
    if isinstance(hit, (int, float)) and hit <= (1.0 / 16.0) + 1e-9:
        return True

    return False
