"""
Benchmark for the synthetic k-th position selection task.

Consumes the payload returned by `task.evaluate` and produces a flat dict of
scalar metrics. See README.md for the full contract.
"""

import math

VERSION = 1

# Pipeline-only hook: every attempt runs on the GPU.
GPU_REQUIREMENT = 1

# Canonical sequence length (fixed; see README.md). Used for the uniform
# baseline (1/L) and the entropy normaliser (log L). The payload does not carry
# L because it is a fixed property of the canonical condition.
L = 32

# Expected sweep positions (length 8). Kept here only for validation; the
# metrics are emitted from whatever records the payload actually contains.
SWEEP_K = [0, 4, 8, 12, 16, 20, 24, 28]


def _k_key(prefix: str, k: int) -> str:
    """Slice key name, e.g. ('kth_select_accuracy', 8) -> 'kth_select_accuracy_k_8'."""
    return f"{prefix}_k_{int(k)}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), canonical_k (int), sweep (list[record]),
        model_name (str), dataset (str).

    Each sweep record: {k, attn_at_k, attn_entropy, attn_max_pos, batch_size}.
    """
    # --- Input validation ---
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_k", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}"
        )
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    canonical_k = payload["canonical_k"]
    if not isinstance(canonical_k, (int, float)):
        raise ValueError(
            f"payload['canonical_k'] must be a number, got {canonical_k!r}"
        )
    canonical_k = int(canonical_k)

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    log_L = math.log(L) if L > 1 else 1.0

    metrics: dict[str, float | int] = {"version": VERSION}

    position_errors: list[float] = []
    seen_canonical = False

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        for field in ("k", "attn_at_k", "attn_entropy", "attn_max_pos"):
            if field not in rec:
                raise KeyError(f"sweep record missing {field!r}")

        k = int(rec["k"])
        attn_at_k = float(rec["attn_at_k"])
        entropy = float(rec["attn_entropy"])
        max_pos = float(rec["attn_max_pos"])

        # Normalised concentration: 1 = delta spike, 0 = uniform.
        if log_L > 1e-12:
            sharpness = 1.0 - entropy / log_L
        else:
            sharpness = 0.0
        sharpness = max(0.0, min(1.0, sharpness))

        metrics[_k_key("kth_select_accuracy", k)] = attn_at_k
        metrics[_k_key("kth_select_sharpness", k)] = sharpness

        position_errors.append(abs(max_pos - k))

        if k == canonical_k:
            metrics["kth_select_accuracy_canonical"] = attn_at_k
            metrics["kth_select_sharpness_canonical"] = sharpness
            seen_canonical = True

    # If the canonical k was not present in the sweep, fall back to 0.0 so the
    # headline keys always exist (defensive; the canonical k should be present).
    if not seen_canonical:
        metrics.setdefault("kth_select_accuracy_canonical", 0.0)
        metrics.setdefault("kth_select_sharpness_canonical", 0.0)

    # Average positional error across the sweep (tokens). Smaller = better.
    if position_errors:
        metrics["kth_select_position_bias"] = float(
            sum(position_errors) / len(position_errors)
        )
    else:
        metrics["kth_select_position_bias"] = float(L)

    # Uniform-attention baseline and lift over it.
    baseline_acc = 1.0 / L if L > 0 else 0.0
    metrics["linear_baseline_accuracy_canonical"] = float(baseline_acc)
    metrics["lift_over_linear_baseline_canonical"] = float(
        metrics["kth_select_accuracy_canonical"] - baseline_acc
    )

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
    # position is mechanically degenerate — selection did nothing.
    acc = metrics.get("kth_select_accuracy_canonical")
    baseline = metrics.get("linear_baseline_accuracy_canonical")
    if isinstance(acc, (int, float)) and isinstance(baseline, (int, float)):
        if acc <= baseline:
            return True

    return False
