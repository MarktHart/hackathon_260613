import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1

_EPS = 1e-9


def _slice_key(prefix: str, shift: int) -> str:
    """Slice key name, e.g. ('local_align', -2) -> 'local_align_shift_m2'."""
    if shift < 0:
        tag = f"m{abs(shift)}"
    elif shift > 0:
        tag = f"p{shift}"
    else:
        tag = "0"
    return f"{prefix}_shift_{tag}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), canonical_shift (int), sequence_length (int),
        vocab_size (int), batch_size (int), measured_head (int),
        sweep (list[record]).

    Each sweep record:
        {shift, mean_max_attn_to_target, mean_entropy, frac_peak_on_target}.
    """
    # --- Input validation ---
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    required_keys = [
        "version", "canonical_shift", "sequence_length", "measured_head", "sweep",
    ]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}"
        )
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    seq_len = payload["sequence_length"]
    if not isinstance(seq_len, (int, float)) or seq_len <= 1:
        raise ValueError(
            f"payload['sequence_length'] must be a number > 1, got {seq_len!r}"
        )
    seq_len = int(seq_len)

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    canonical_shift = int(payload["canonical_shift"])

    # --- Index records by shift ---
    by_shift: dict[int, dict] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        if "shift" not in rec:
            raise KeyError("sweep record missing 'shift'")
        by_shift[int(rec["shift"])] = rec

    metrics: dict[str, float | int] = {"version": VERSION}

    # --- Per-slice values ---
    for shift, rec in sorted(by_shift.items()):
        align = float(rec.get("mean_max_attn_to_target", 0.0))
        ent = float(rec.get("mean_entropy", 0.0))
        peak = float(rec.get("frac_peak_on_target", 0.0))
        metrics[_slice_key("local_align", shift)] = align
        metrics[_slice_key("local_align_entropy", shift)] = ent
        metrics[_slice_key("local_align_peak", shift)] = peak

    # --- Canonical (shift = -1 by default) ---
    canon = by_shift.get(canonical_shift, {})
    canonical_align = float(canon.get("mean_max_attn_to_target", 0.0))
    canonical_peak = float(canon.get("frac_peak_on_target", 0.0))
    canonical_entropy = float(canon.get("mean_entropy", 0.0))

    metrics["local_align_canonical"] = canonical_align
    metrics["local_align_peak_canonical"] = canonical_peak
    metrics["local_align_entropy_canonical"] = canonical_entropy

    # --- Headline: robustness vs strongest off-diagonal distractor ---
    distractor_shifts = [0, 1, 2]
    distractor_vals = [
        float(by_shift.get(s, {}).get("mean_max_attn_to_target", 0.0))
        for s in distractor_shifts
        if s in by_shift
    ]
    max_distractor = max(distractor_vals) if distractor_vals else 0.0
    if max_distractor > _EPS:
        metrics["local_align_robustness"] = canonical_align / max_distractor
    else:
        metrics["local_align_robustness"] = 0.0

    # --- Baselines (same canonical condition) ---
    uniform_baseline = 1.0 / max(seq_len - 1, 1)
    metrics["linear_baseline_canonical"] = float(uniform_baseline)
    metrics["lift_over_uniform_canonical"] = canonical_align - float(uniform_baseline)
    metrics["random_baseline_peak_canonical"] = 1.0 / float(seq_len)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that does not even beat uniform attention at the canonical
    # condition is mechanically degenerate; skip the (expensive) jury.
    align = metrics.get("local_align_canonical")
    baseline = metrics.get("linear_baseline_canonical")
    if isinstance(align, (int, float)) and isinstance(baseline, (int, float)):
        if align <= baseline:
            return True

    return False
