import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _len_key(prefix: str, length: int) -> str:
    """Slice key name, e.g. ('sort_accuracy', 8) -> 'sort_accuracy_len_8'."""
    return f"{prefix}_len_{int(length)}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1), canonical_length (int),
        sweep_lengths (list[int]), sweep (list[record]).

    Each sweep record:
        {length, sort_accuracy, target_mass, output_sortedness,
         unsorted_sortedness, uniform_accuracy, n_sequences}.

    Directionality: every metric here is bigger-is-better.
    """
    # --- Input validation ---
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ("version", "canonical_length", "sweep"):
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)) or len(sweep) == 0:
        raise ValueError("payload['sweep'] must be a non-empty list")

    canonical_length = int(payload["canonical_length"])

    metrics: dict[str, float | int] = {"version": VERSION}

    by_length: dict[int, dict] = {}
    accuracies: list[tuple[int, float]] = []

    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        if "length" not in rec:
            raise KeyError("sweep record missing 'length'")
        L = int(rec["length"])
        by_length[L] = rec

        acc = float(rec.get("sort_accuracy", 0.0))
        tmass = float(rec.get("target_mass", 0.0))
        out_sorted = float(rec.get("output_sortedness", 0.0))
        unsorted = float(rec.get("unsorted_sortedness", 0.0))

        metrics[_len_key("sort_accuracy", L)] = acc
        metrics[_len_key("target_mass", L)] = tmass
        metrics[_len_key("output_sortedness", L)] = out_sorted
        metrics[_len_key("unsorted_baseline_sortedness", L)] = unsorted

        accuracies.append((L, acc))

    # --- Canonical condition ---
    canon = by_length.get(canonical_length, {})
    metrics["sort_accuracy_canonical"] = float(canon.get("sort_accuracy", 0.0))
    metrics["output_sortedness_canonical"] = float(canon.get("output_sortedness", 0.0))
    metrics["target_mass_canonical"] = float(canon.get("target_mass", 0.0))

    uniform_acc_canonical = float(canon.get("uniform_accuracy", 0.0))
    metrics["uniform_baseline_accuracy_canonical"] = uniform_acc_canonical

    metrics["lift_over_unsorted_canonical"] = (
        metrics["output_sortedness_canonical"]
        - float(canon.get("unsorted_sortedness", 0.0))
    )

    # --- Headline: sort_robustness ---
    # Accuracy retained at the longest sequence relative to the shortest. A head
    # that truly implements sorting holds up as L grows; a positional shortcut
    # decays. Ratio in [0, 1]; bigger is better.
    accuracies.sort(key=lambda t: t[0])
    acc_short = accuracies[0][1]
    acc_long = accuracies[-1][1]
    if acc_short > 1e-12:
        robustness = acc_long / acc_short
    else:
        robustness = 0.0
    metrics["sort_robustness"] = float(max(0.0, min(1.0, robustness)))

    # Mean accuracy across the whole sweep — overall quality summary.
    metrics["sort_accuracy_mean"] = float(
        sum(a for _, a in accuracies) / len(accuracies)
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

    # An attempt that cannot clearly beat uniform/random attention at the
    # canonical condition is mechanically degenerate. Uniform argmax accuracy is
    # ~1/L; require a margin above it before spending the jury.
    acc = metrics.get("sort_accuracy_canonical")
    uni = metrics.get("uniform_baseline_accuracy_canonical")
    if isinstance(acc, (int, float)) and isinstance(uni, (int, float)):
        if acc <= max(uni * 2.0, 0.05):
            return True

    return False
