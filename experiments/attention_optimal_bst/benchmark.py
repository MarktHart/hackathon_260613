"""Benchmark for the attention_optimal_bst goal.

Consumes the payload returned by task.evaluate(model_fn) and produces a flat
dict of scalar metrics. Pure Python; deterministic; side-effect free.
"""

import math

VERSION = 1

# Pipeline-only hook: how many GPU slots the attempt subprocess needs.
GPU_REQUIREMENT = 1


def _pathlen_key(prefix: str, path_len: int) -> str:
    """Slice key name, e.g. ('bst_search_accuracy', 3) -> 'bst_search_accuracy_pathlen_3'."""
    return f"{prefix}_pathlen_{int(path_len)}"


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute flat scalar metrics from the payload returned by task.evaluate().

    Expected payload keys:
        version (int == 1),
        canonical_condition (dict with n_keys, ...),
        sweep (list[record]),
        aggregated (dict with best_head, mean_path_attention,
                    perfect_episodes, total_episodes, mean_path_completion_rate).

    Each sweep record:
        {query_key, optimal_path, attn_to_path, path_length, head_idx}.
    """
    # --- Input validation ---
    if not isinstance(payload, dict):
        raise ValueError(f"payload must be a dict, got {type(payload).__name__}")

    for k in ["version", "canonical_condition", "sweep", "aggregated"]:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k!r}")

    version = payload["version"]
    if not isinstance(version, int):
        raise ValueError(f"payload['version'] must be int, got {type(version).__name__}")
    if version != VERSION:
        raise ValueError(f"Unsupported payload version: {version}. Expected {VERSION}.")

    cond = payload["canonical_condition"]
    if not isinstance(cond, dict):
        raise ValueError("payload['canonical_condition'] must be a dict")
    n_keys = int(cond.get("n_keys", 15))
    if n_keys <= 0:
        raise ValueError(f"canonical_condition['n_keys'] must be positive, got {n_keys}")

    sweep = payload["sweep"]
    if not isinstance(sweep, (list, tuple)):
        raise ValueError("payload['sweep'] must be a list")

    agg = payload["aggregated"]
    if not isinstance(agg, dict):
        raise ValueError("payload['aggregated'] must be a dict")

    total = int(agg.get("total_episodes", len(sweep)))
    if total <= 0:
        # Empty batch — return well-defined zeros rather than dividing by zero.
        total = max(total, 0)

    perfect = int(agg.get("perfect_episodes", 0))
    mean_path_attention = float(agg.get("mean_path_attention", 0.0))
    best_head = int(agg.get("best_head", 0))

    metrics: dict[str, float | int] = {"version": VERSION}

    # --- Headline + canonical metrics ---
    if total > 0:
        accuracy = perfect / total
    else:
        accuracy = 0.0

    metrics["bst_search_accuracy_canonical"] = float(accuracy)
    metrics["bst_mean_path_attention_canonical"] = float(mean_path_attention)
    metrics["bst_best_head_canonical"] = best_head

    # Path completion rate: prefer the pre-aggregated value, else derive from sweep.
    if "mean_path_completion_rate" in agg:
        completion = float(agg["mean_path_completion_rate"])
    else:
        completion = _completion_from_sweep(sweep)
    metrics["bst_path_completion_rate_canonical"] = float(completion)

    # --- Per-slice values: accuracy by optimal-path length ---
    # Group episodes by path length; for each, what fraction are "perfect"
    # (best head puts >0.5 on every path node)?
    by_len_perfect: dict[int, int] = {}
    by_len_total: dict[int, int] = {}
    for rec in sweep:
        if not isinstance(rec, dict):
            raise ValueError("Each sweep record must be a dict")
        plen = int(rec.get("path_length", len(rec.get("optimal_path", []))))
        attn_on_path = rec.get("attn_to_path", []) or []
        by_len_total[plen] = by_len_total.get(plen, 0) + 1
        if plen > 0:
            n_above = sum(1 for a in attn_on_path if float(a) > 0.5)
            is_perfect = (n_above == plen)
        else:
            is_perfect = True  # zero-length path is vacuously satisfied
        if is_perfect:
            by_len_perfect[plen] = by_len_perfect.get(plen, 0) + 1

    for plen in sorted(by_len_total):
        denom = by_len_total[plen]
        num = by_len_perfect.get(plen, 0)
        slice_acc = (num / denom) if denom > 0 else 0.0
        metrics[_pathlen_key("bst_search_accuracy", plen)] = float(slice_acc)

    # --- Reference baseline: uniform attention over the n key nodes ---
    # Mass per node is 1/n_keys < 0.5 for any realistic n_keys, so no path
    # position ever clears the 0.5 threshold -> baseline accuracy is 0.
    baseline_mass = 1.0 / n_keys
    baseline_accuracy = 0.0 if baseline_mass <= 0.5 else 1.0
    metrics["linear_baseline_bst_search_accuracy_canonical"] = float(baseline_accuracy)
    metrics["linear_baseline_mean_path_attention_canonical"] = float(baseline_mass)

    metrics["lift_over_linear_baseline_bst_search_accuracy"] = float(
        accuracy - baseline_accuracy
    )
    metrics["lift_over_linear_baseline_mean_path_attention"] = float(
        mean_path_attention - baseline_mass
    )

    return metrics


def _completion_from_sweep(sweep) -> float:
    """Mean over episodes of (positions with attn > 0.5) / path_length."""
    rates = []
    for rec in sweep:
        plen = int(rec.get("path_length", len(rec.get("optimal_path", []))))
        attn_on_path = rec.get("attn_to_path", []) or []
        if plen == 0:
            rates.append(1.0)
            continue
        n_above = sum(1 for a in attn_on_path if float(a) > 0.5)
        rates.append(n_above / plen)
    if not rates:
        return 0.0
    return sum(rates) / len(rates)


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: True if metrics are clearly degenerate, to skip the jury.
    Never returns True for a borderline-but-real result.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Worse than (or equal to) the no-mechanism uniform baseline on the core
    # quantity: attention mass on path nodes that does not beat 1/n_keys means
    # the attempt isn't navigating the tree at all.
    mean_attn = metrics.get("bst_mean_path_attention_canonical")
    baseline = metrics.get("linear_baseline_mean_path_attention_canonical")
    if isinstance(mean_attn, (int, float)) and isinstance(baseline, (int, float)):
        # Small tolerance so a genuinely no-mechanism (uniform) attempt is still
        # caught despite float32 vs float64 rounding of 1/n_keys. A real method
        # that navigates the tree clears the baseline by far more than this.
        if mean_attn <= baseline * (1.0 + 1e-4):
            return True

    return False
