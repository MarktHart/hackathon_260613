import math

VERSION = 1


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute metrics from the payload returned by task.evaluate().
    """
    # Validate payload structure
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a dict")
    if "version" not in payload:
        raise KeyError("Payload missing 'version'")
    if "config" not in payload:
        raise KeyError("Payload missing 'config'")
    if "sweep" not in payload:
        raise KeyError("Payload missing 'sweep'")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Sweep must be a non-empty list")

    # Validate each sweep record
    required_fields = ["num_heads", "mean_attn_within", "mean_attn_between", "retrieval_acc"]
    for i, record in enumerate(sweep):
        for f in required_fields:
            if f not in record:
                raise KeyError(f"Sweep record {i} missing field: {f!r}")

    # Find canonical record (num_heads=4)
    canonical_record = None
    for record in sweep:
        if record["num_heads"] == 4:
            canonical_record = record
            break

    if canonical_record is None:
        raise KeyError("Canonical sweep record (num_heads=4) not found")

    metrics = {}
    metrics["version"] = payload["version"]

    # Per-slice metrics
    for record in sweep:
        h = record["num_heads"]
        bipartite_score = record["mean_attn_between"] - record["mean_attn_within"]
        metrics[f"bipartite_score_num_heads_{h}"] = bipartite_score
        metrics[f"retrieval_acc_num_heads_{h}"] = record["retrieval_acc"]

        # Linear baseline: uniform attention within valid targets (no softmax structure)
        # For bipartite task, uniform cross-group = 1/(group_size) per target, within = 0
        # But linear baseline here means: what if we just used raw dot products without softmax?
        # Simplified: uniform over all positions gives 1/seq_len for everything.
        # Bipartite score for uniform = 0. But we compute a more meaningful baseline:
        # /seq_len for between, seq_len/2 / seq_len = 0.5 for within? No.
        # Actually for uniform attention: mean_between = group_size/seq_len = 0.5, mean_within = 0.5 -> score = 0
        # So linear baseline is 0. But we include it for completeness.
        seq_len = payload["config"]["group_size"] * 2
        group_size = payload["config"]["group_size"]
        uniform_between = group_size / seq_len  # 0.5
        uniform_within = group_size / seq_len   # 0.5
        metrics[f"linear_baseline_bipartite_score_num_heads_{h}"] = uniform_between - uniform_within  # = 0

    # Canonical headline metric
    canonical_score = canonical_record["mean_attn_between"] - canonical_record["mean_attn_within"]
    metrics["bipartite_score_canonical"] = canonical_score

    # Robustness: min/max ratio across sweep (clamped to [0,1])
    scores = [metrics[f"bipartite_score_num_heads_{r['num_heads']}"] for r in sweep]
    min_score = min(scores)
    max_score = max(scores)
    if max_score > 0:
        robustness = max(0.0, min_score / max_score)
    else:
        robustness = 0.0
    metrics["bipartite_robustness"] = robustness

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: return True if metrics indicate a clearly failed attempt.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Headline metric should beat linear baseline (which is 0) by a meaningful margin
    canonical = metrics.get("bipartite_score_canonical")
    if isinstance(canonical, (int, float)):
        if canonical <= 0.01:  # barely positive or negative
            return True

    # Retrieval accuracy should be above random (1/num_targets ≈ 1/8 = 0.125)
    # but random_model_fn will give ~0.5 uniform, so we check for collapse
    for key, val in metrics.items():
        if key.startswith("retrieval_acc_") and isinstance(val, (int, float)):
            if val < 0.1:  # worse than random guessing
                return True

    return False