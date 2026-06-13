import math
from typing import Dict, Any


VERSION = 1


def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    """
    Compute metrics from the payload produced by task.evaluate.
    """
    # Validate required keys
    required_keys = ["version", "modulus", "layer_index", "head_index", "d_head", "sweep"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k}")

    version = payload["version"]
    if version != VERSION:
        raise ValueError(f"Payload version {version} != benchmark VERSION {VERSION}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Payload 'sweep' must be a non-empty list")

    d_head = payload["d_head"]
    if not isinstance(d_head, int) or d_head <= 0:
        raise ValueError("d_head must be a positive integer")

    # Extract per-frequency values
    alignments = []
    phase_errors = []
    explained_vars = []
    freq_to_alignment = {}
    freq_to_phase = {}

    for rec in sweep:
        if not all(k in rec for k in ("frequency", "alignment", "phase_error", "explained_variance")):
            raise KeyError("Each sweep record must have frequency, alignment, phase_error, explained_variance")
        k = rec["frequency"]
        align = float(rec["alignment"])
        phase = float(rec["phase_error"])
        ev = float(rec["explained_variance"])

        # Validate ranges
        if not (0.0 <= align <= 1.0):
            raise ValueError(f"alignment for k={k} out of range [0,1]: {align}")
        if not (0.0 <= phase <= math.pi):
            raise ValueError(f"phase_error for k={k} out of range [0,π]: {phase}")
        if not (0.0 <= ev <= 1.0):
            raise ValueError(f"explained_variance for k={k} out of range [0,1]: {ev}")

        alignments.append(align)
        phase_errors.append(phase)
        explained_vars.append(ev)
        freq_to_alignment[k] = align
        freq_to_phase[k] = phase

    # Headline summary metric: mean alignment across all frequencies
    fourier_alignment_canonical = float(np_mean(alignments))
    phase_error_canonical = float(np_mean(phase_errors))
    explained_variance_canonical = float(np_mean(explained_vars))

    # Random baseline: expected alignment for random vectors in d_head dimensions
    # For random unit vectors in R^d projected onto 2D subspace, E[cos^2] = 2/d
    # Alignment uses cosine similarity (not squared), so approximate baseline
    random_baseline_alignment = 2.0 / d_head

    # Lift over baseline
    lift_over_random_alignment = fourier_alignment_canonical - random_baseline_alignment

    # Superposition robustness: ratio of min to max alignment across frequencies
    min_align = min(alignments)
    max_align = max(alignments)
    if max_align > 0:
        superposition_robustness = float(min_align / max_align)
    else:
        superposition_robustness = 0.0

    # Per-slice metrics
    metrics = {
        "version": VERSION,
        "fourier_alignment_canonical": fourier_alignment_canonical,
        "phase_error_canonical": phase_error_canonical,
        "explained_variance_canonical": explained_variance_canonical,
        "random_baseline_alignment": random_baseline_alignment,
        "lift_over_random_alignment": lift_over_random_alignment,
        "superposition_robustness": superposition_robustness,
    }

    for k, align in freq_to_alignment.items():
        metrics[f"fourier_alignment_freq_{k:02d}"] = float(align)
    for k, phase in freq_to_phase.items():
        metrics[f"phase_error_freq_{k:02d}"] = float(phase)

    return metrics


def np_mean(xs):
    """Helper to avoid numpy dependency in benchmark.py."""
    return sum(xs) / len(xs) if xs else 0.0


def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    """
    Return True ONLY for clearly degenerate results, to short-circuit the jury.

    This must never fire on a borderline-but-real mechanism. In particular we do
    NOT threshold on mean explained_variance or mean phase_error: a clean
    single/few-frequency head legitimately concentrates its variance on a couple
    of frequencies, so those means are small *by construction*. We only catch
    math failures and at-or-below-chance alignment.
    """
    # NaN / inf anywhere -> broken.
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Headline alignment at or below the analytic chance baseline. The baseline
    # (2/d_head) is conservative (empirically random noise scores above it), so
    # tripping this means the head carries essentially no Fourier structure.
    alignment = metrics.get("fourier_alignment_canonical")
    baseline = metrics.get("random_baseline_alignment")
    if isinstance(alignment, (int, float)) and isinstance(baseline, (int, float)):
        if alignment <= baseline:
            return True

    return False


# Optional: GPU requirement for large model activation extraction
GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU