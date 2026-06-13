import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _len_key(prefix: str, seq_len: int) -> str:
    """Slice key name, e.g. ('reverse_accuracy', 16) -> 'reverse_accuracy_len_16'."""
    return f"{prefix}_len_{int(seq_len)}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), vocab_size (int), seq_len_sweep (list[int]),
        canonical_idx (int), sweep (list[record]).

    Each sweep record:
        {seq_len, accuracy, mirror_attn_mass,
         identity_baseline_accuracy, num_sequences}.
    """
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ["version", "vocab_size", "seq_len_sweep", "canonical_idx", "sweep"]:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    vocab_size = payload["vocab_size"]
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"payload['vocab_size'] must be a positive int, got {vocab_size!r}")

    seq_len_sweep = payload["seq_len_sweep"]
    if not isinstance(seq_len_sweep, (list, tuple)) or len(seq_len_sweep) == 0:
        raise ValueError("payload['seq_len_sweep'] must be a non-empty list")
    seq_len_sweep = [int(s) for s in seq_len_sweep]

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) != len(seq_len_sweep):
        raise ValueError("payload['sweep'] must be a list of same length as seq_len_sweep")

    canonical_idx = payload["canonical_idx"]
    if not isinstance(canonical_idx, int) or not (0 <= canonical_idx < len(sweep)):
        raise ValueError(
            f"payload['canonical_idx'] {canonical_idx!r} out of range for sweep "
            f"length {len(sweep)}"
        )

    # --- Index records by sequence length ---
    by_len: dict[int, dict] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("each sweep record must be a dict")
        for rk in ["seq_len", "accuracy", "mirror_attn_mass", "identity_baseline_accuracy"]:
            if rk not in rec:
                raise KeyError(f"sweep record missing {rk!r}")
        by_len[int(rec["seq_len"])] = rec

    metrics: dict[str, float | int] = {"version": VERSION}

    accuracies: list[float] = []
    for seq_len in seq_len_sweep:
        rec = by_len.get(seq_len, {})
        acc = float(rec.get("accuracy", 0.0))
        mass = float(rec.get("mirror_attn_mass", 0.0))
        ident = float(rec.get("identity_baseline_accuracy", 0.0))

        metrics[_len_key("reverse_accuracy", seq_len)] = acc
        metrics[_len_key("mirror_attn_mass", seq_len)] = mass
        metrics[_len_key("identity_baseline_accuracy", seq_len)] = ident
        accuracies.append(acc)

    # --- Canonical condition ---
    canonical_seq_len = seq_len_sweep[canonical_idx]
    acc_canonical = float(metrics.get(_len_key("reverse_accuracy", canonical_seq_len), 0.0))
    mass_canonical = float(metrics.get(_len_key("mirror_attn_mass", canonical_seq_len), 0.0))

    metrics["reverse_accuracy_canonical"] = acc_canonical
    metrics["mirror_attn_mass_canonical"] = mass_canonical

    # Random-guess reference (no mechanism at all).
    random_baseline = 1.0 / vocab_size
    metrics["random_baseline_accuracy"] = random_baseline
    metrics["lift_over_random_canonical"] = acc_canonical - random_baseline

    # --- Headline: length-generalisation robustness ---
    # Worst accuracy at lengths BEYOND the canonical length, relative to the
    # canonical accuracy. 1.0 = extrapolates perfectly; 0.0 = collapses.
    gen_acc = [
        float(metrics.get(_len_key("reverse_accuracy", s), 0.0))
        for s in seq_len_sweep if s > canonical_seq_len
    ]
    if gen_acc and acc_canonical > 1e-12:
        robustness = min(gen_acc) / acc_canonical
    else:
        robustness = 0.0
    metrics["length_generalization_robustness"] = float(max(0.0, min(1.0, robustness)))

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # An attempt that does not even beat random guessing at the canonical length
    # has not learned reversal at all.
    acc = metrics.get("reverse_accuracy_canonical")
    baseline = metrics.get("random_baseline_accuracy")
    if isinstance(acc, (int, float)) and isinstance(baseline, (int, float)):
        if acc <= baseline * 1.05:
            return True

    return False
