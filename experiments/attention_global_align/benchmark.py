"""Benchmark: attention_global_align.

Pure-Python, deterministic scoring of the payload produced by task.evaluate().
No imports from any attempt directory; no I/O; no time-dependent values.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _fmt(c: float) -> str:
    """Slice float -> key fragment: 0.0->'0p0', 0.25->'0p25', 0.5->'0p5', 1.0->'1p0'."""
    s = "%g" % float(c)
    if "." not in s and "e" not in s:
        s += ".0"
    return s.replace(".", "p").replace("-", "neg")


def _slice_key(prefix: str, c: float) -> str:
    return f"{prefix}_dist_{_fmt(c)}"


def score(payload: dict) -> dict[str, float | int]:
    """Compute flat scalar metrics from the task.evaluate() payload.

    Expected payload keys:
        version (int == 1), d (int), seq_len (int),
        canonical_distractor_cos (float), distractor_cos_sweep (list[float]),
        sweep (list[record]), uniform_baseline (list[record]).

    Each sweep record: {distractor_cos, global_alignment, distractor_mass,
                        target_margin, n_seqs}.
    Each uniform_baseline record: {distractor_cos, global_alignment, n_seqs}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required = ["version", "d", "seq_len", "canonical_distractor_cos",
                "distractor_cos_sweep", "sweep", "uniform_baseline"]
    for k in required:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    cos_sweep = payload["distractor_cos_sweep"]
    if not isinstance(cos_sweep, (list, tuple)) or len(cos_sweep) == 0:
        raise ValueError("payload['distractor_cos_sweep'] must be a non-empty list")
    cos_sweep = [float(c) for c in cos_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(cos_sweep):
        raise ValueError("payload['sweep'] must be a list as long as distractor_cos_sweep")

    baseline = payload["uniform_baseline"]
    if not isinstance(baseline, (list, tuple)) or len(baseline) != len(cos_sweep):
        raise ValueError(
            "payload['uniform_baseline'] must be a list as long as distractor_cos_sweep"
        )

    def _index(records, what):
        out = {}
        for rec in records:
            if not isinstance(rec, dict):
                raise ValueError(f"Each {what} record must be a dict")
            if "distractor_cos" not in rec:
                raise KeyError(f"{what} record missing 'distractor_cos'")
            out[round(float(rec["distractor_cos"]), 6)] = rec
        return out

    sweep_by = _index(sweep, "sweep")
    base_by = _index(baseline, "uniform_baseline")

    metrics: dict[str, float | int] = {"version": VERSION}

    align_vals: list[float] = []
    for c in cos_sweep:
        key = round(c, 6)
        srec = sweep_by.get(key, {})
        brec = base_by.get(key, {})

        align = float(srec.get("global_alignment", 0.0))
        dmass = float(srec.get("distractor_mass", 0.0))
        margin = float(srec.get("target_margin", 0.0))
        b_align = float(brec.get("global_alignment", 0.0))

        metrics[_slice_key("global_alignment", c)] = align
        metrics[_slice_key("distractor_mass", c)] = dmass
        metrics[_slice_key("target_margin", c)] = margin
        metrics[_slice_key("uniform_baseline_alignment", c)] = b_align

        align_vals.append(align)

    # --- Canonical condition ---
    canonical = float(payload["canonical_distractor_cos"])
    canonical_align = float(metrics.get(_slice_key("global_alignment", canonical), 0.0))
    canonical_base = float(metrics.get(_slice_key("uniform_baseline_alignment", canonical), 0.0))
    metrics["global_alignment_canonical"] = canonical_align
    metrics["lift_over_uniform_canonical"] = canonical_align - canonical_base

    # --- Mean alignment across the whole sweep ---
    metrics["global_alignment_mean"] = (
        float(sum(align_vals) / len(align_vals)) if align_vals else 0.0
    )

    # --- Headline: global_alignment_robustness ---
    # Alignment retained under maximum interference (last slice) relative to the
    # no-interference anchor (first slice), clipped to [0, 1].
    align_low = float(metrics.get(_slice_key("global_alignment", cos_sweep[0]), 0.0))
    align_high = float(metrics.get(_slice_key("global_alignment", cos_sweep[-1]), 0.0))
    if align_low > 1e-12:
        robustness = align_high / align_low
    else:
        robustness = 0.0
    metrics["global_alignment_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """True if metrics are clearly degenerate, to skip the (expensive) jury.

    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that cannot beat the uniform baseline at the canonical
    # condition has no working retrieval mechanism to evaluate.
    align = metrics.get("global_alignment_canonical")
    base = metrics.get("uniform_baseline_alignment_dist_0p5")
    if isinstance(align, (int, float)) and isinstance(base, (int, float)):
        if align <= base:
            return True

    return False
