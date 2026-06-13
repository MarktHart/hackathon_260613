import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _head_mean_alignment(record: dict) -> float:
    """Average of per-head `mean_alignment` for one sweep record."""
    heads = record.get("head_alignments", [])
    if not heads:
        return 0.0
    vals = [float(h.get("mean_alignment", 0.0)) for h in heads]
    return sum(vals) / len(vals)


def _head_max_alignment(record: dict) -> float:
    """Best-head `max_alignment` for one sweep record."""
    heads = record.get("head_alignments", [])
    if not heads:
        return 0.0
    return max(float(h.get("max_alignment", 0.0)) for h in heads)


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1),
        config (dict with seq_len, vocab_size, batch_size, perm_types,
                n_heads, n_layers),
        sweep (list of records, one per permutation type), each:
            {perm_type, head_alignments: [{head_idx, mean_alignment,
             max_alignment, alignment_per_pos}], layer_idx},
        random_baseline (dict perm_type -> {mean_alignment, ...}).
    """
    # --- Input validation ---
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "config", "sweep", "random_baseline"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(
            f"payload['version'] must be int, got {type(version).__name__}"
        )
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    config = payload["config"]
    if not isinstance(config, dict):
        raise ValueError("payload['config'] must be a dict")
    seq_len = config.get("seq_len")
    if not isinstance(seq_len, int) or seq_len <= 0:
        raise ValueError(f"config['seq_len'] must be a positive int, got {seq_len!r}")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")

    # --- Index sweep records by perm_type (canonical layer is 0) ---
    by_perm: dict[str, dict] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        if "perm_type" not in rec:
            raise KeyError("sweep record missing 'perm_type'")
        if rec.get("layer_idx", 0) != 0:
            continue
        # Keep the first record per perm_type at layer 0.
        by_perm.setdefault(str(rec["perm_type"]), rec)

    metrics: dict[str, float | int] = {"version": VERSION}

    random_baseline_alignment = 1.0 / seq_len
    metrics["random_baseline_alignment"] = float(random_baseline_alignment)

    # --- Per-slice metrics: one mean-alignment per permutation type present ---
    per_perm_alignment: dict[str, float] = {}
    for perm_type, rec in by_perm.items():
        align = _head_mean_alignment(rec)
        per_perm_alignment[perm_type] = align
        metrics[f"anagram_alignment_{perm_type}"] = float(align)

    # --- Canonical: random permutation, mean over heads at layer 0 ---
    canonical_align = float(per_perm_alignment.get("random", 0.0))
    metrics["anagram_alignment_canonical"] = canonical_align

    canonical_rec = by_perm.get("random")
    metrics["anagram_alignment_max_canonical"] = (
        float(_head_max_alignment(canonical_rec)) if canonical_rec else 0.0
    )

    metrics["lift_over_random_canonical"] = (
        canonical_align - random_baseline_alignment
    )

    # --- Headline / consistency: alignment_robustness in [0, 1] ---
    vals = list(per_perm_alignment.values())
    if vals and max(vals) > 1e-12:
        metrics["alignment_robustness"] = float(
            max(0.0, min(1.0, min(vals) / max(vals)))
        )
    else:
        metrics["alignment_robustness"] = 0.0

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that does not even beat the uniform-attention baseline at the
    # canonical (random-permutation) condition is mechanically degenerate.
    canonical = metrics.get("anagram_alignment_canonical")
    baseline = metrics.get("random_baseline_alignment")
    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        if canonical <= baseline:
            return True

    return False
