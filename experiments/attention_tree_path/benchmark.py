import math


VERSION = 1


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute metrics from payload.
    Payload must match the contract in README.md.
    """
    # Validate version
    if payload.get("version") != VERSION:
        raise ValueError(f"Payload version {payload.get('version')} != benchmark VERSION {VERSION}")

    # Validate required keys
    required_keys = ["config", "sweep", "head_slice"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k}")

    sweep = payload["sweep"]
    head_slice = payload["head_slice"]
    config = payload["config"]

    # Validate sweep length
    if len(sweep) != 6:
        raise ValueError(f"Expected 6 sweep records, got {len(sweep)}")

    # Validate head_slice length
    if len(head_slice) != 4:
        raise ValueError(f"Expected 4 head_slice records, got {len(head_slice)}")

    # Helper to find sweep record
    def get_sweep(depth: int, path_rule: str) -> dict:
        for rec in sweep:
            if rec["depth"] == depth and rec["path_rule"] == path_rule:
                return rec
        raise KeyError(f"Sweep record not found: depth={depth}, path_rule={path_rule}")

    # Extract per-slice values
    s_d2_a1 = get_sweep(2, "ancestor_1")
    s_d3_a1 = get_sweep(3, "ancestor_1")
    s_d4_a1 = get_sweep(4, "ancestor_1")
    s_d3_a2 = get_sweep(3, "ancestor_2")
    s_d3_desc = get_sweep(3, "descendant_leftmost")
    s_d3_sib = get_sweep(3, "sibling_next")

    # Headline metric: canonical condition (depth=3, ancestor_1)
    tree_path_canonical = s_d3_a1["correct_attn_mean"]

    # Baseline: uniform attention over other positions
    seq_len = config.get("seq_len", 15)
    linear_baseline_canonical = 1.0 / (seq_len - 1) if seq_len > 1 else 0.0

    # Depth robustness ratio
    depths_vals = [
        s_d2_a1["correct_attn_mean"],
        s_d3_a1["correct_attn_mean"],
        s_d4_a1["correct_attn_mean"],
    ]
    max_d = max(depths_vals)
    min_d = min(depths_vals)
    tree_path_robustness = min_d / max_d if max_d > 0 else 0.0

    # Head slice stats
    head_means = [h["correct_attn_mean"] for h in head_slice]
    tree_path_head_best = max(head_means)
    tree_path_head_worst = min(head_means)
    tree_path_head_gap = tree_path_head_best - tree_path_head_worst

    return {
        "version": VERSION,
        "tree_path_canonical": tree_path_canonical,
        "tree_path_depth_2": s_d2_a1["correct_attn_mean"],
        "tree_path_depth_3": s_d3_a1["correct_attn_mean"],
        "tree_path_depth_4": s_d4_a1["correct_attn_mean"],
        "tree_path_ancestor_2": s_d3_a2["correct_attn_mean"],
        "tree_path_descendant": s_d3_desc["correct_attn_mean"],
        "tree_path_sibling": s_d3_sib["correct_attn_mean"],
        "tree_path_robustness": tree_path_robustness,
        "tree_path_head_best": tree_path_head_best,
        "tree_path_head_worst": tree_path_head_worst,
        "tree_path_head_gap": tree_path_head_gap,
        "linear_baseline_canonical": linear_baseline_canonical,
        "lift_over_baseline": tree_path_canonical - linear_baseline_canonical,
    }


def is_obviously_broken(metrics: dict) -> bool:
    """
    Return True if metrics indicate a fundamentally broken attempt.
    Used by pipeline to skip jury on clear failures.
    """
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Headline metric worse than or equal to baseline (no signal)
    canonical = metrics.get("tree_path_canonical")
    baseline = metrics.get("linear_baseline_canonical")
    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        if canonical <= baseline * 1.01:  # Allow tiny numerical noise
            return True

    # Robustness is NaN or zero (all depths zero)
    robustness = metrics.get("tree_path_robustness")
    if isinstance(robustness, float) and robustness == 0.0:
        return True

    return False


# Optional: GPU requirement for attempts (default 1)
GPU_REQUIREMENT = 1