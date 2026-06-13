import math

VERSION = 1


def _spearman_r(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    
    # Rank with average for ties
    def rankdata(arr: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n and arr[sorted_indices[j]] == arr[sorted_indices[i]]:
                j += 1
            avg_rank = (i + 1 + j) / 2.0
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            i = j
        return ranks
    
    rx = rankdata(x)
    ry = rankdata(y)
    
    # Pearson on ranks
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    
    if var_x == 0 or var_y == 0:
        return 0.0
    
    return cov / math.sqrt(var_x * var_y)


def _pearson_r(x: list[float], y: list[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    
    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    var_x = sum((x[i] - mean_x) ** 2 for i in range(n))
    var_y = sum((y[i] - mean_y) ** 2 for i in range(n))
    
    if var_x == 0 or var_y == 0:
        return 0.0
    
    return cov / math.sqrt(var_x * var_y)


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute metrics from the task payload.
    
    Args:
        payload: Dict from task.evaluate(), must contain:
            - version: int
            - sweep: list of dicts with keys edit_distance, attn_distance_mean, n_pairs
            - linear_baseline: dict with attn_distance_mean (list[float])
    
    Returns:
        Flat dict of metrics.
    """
    # Validate required keys
    required_keys = ["version", "sweep", "linear_baseline"]
    for key in required_keys:
        if key not in payload:
            raise KeyError(f"Payload missing required key: {key}")
    
    version = payload["version"]
    sweep = payload["sweep"]
    baseline = payload["linear_baseline"]
    
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Payload 'sweep' must be a non-empty list")
    
    if "attn_distance_mean" not in baseline:
        raise KeyError("Payload 'linear_baseline' missing 'attn_distance_mean'")
    
    baseline_means = baseline["attn_distance_mean"]
    
    if len(baseline_means) != len(sweep):
        raise ValueError(
            f"Baseline length ({len(baseline_means)}) != sweep length ({len(sweep)})"
        )
    
    # Extract edit distances and attention distances (skip entries with n_pairs == 0)
    edit_dists = []
    attn_dists = []
    baseline_dists = []
    
    for i, entry in enumerate(sweep):
        n_pairs = entry.get("n_pairs", 0)
        if n_pairs > 0:
            edit_dists.append(float(entry["edit_distance"]))
            attn_dists.append(float(entry["attn_distance_mean"]))
            baseline_dists.append(float(baseline_means[i]))
    
    if len(edit_dists) < 2:
        # Not enough data points for correlation
        headline_corr = 0.0
        baseline_corr = 0.0
    else:
        headline_corr = _spearman_r(edit_dists, attn_dists)
        baseline_corr = _spearman_r(edit_dists, baseline_dists)
    
    pearson_corr = _pearson_r(edit_dists, attn_dists) if len(edit_dists) >= 2 else 0.0
    
    # Build per-slice metrics
    metrics = {
        "version": version,
        "edit_distance_correlation": headline_corr,
        "edit_distance_correlation_pearson": pearson_corr,
        "linear_baseline_correlation": baseline_corr,
        "lift_over_baseline": headline_corr - baseline_corr,
    }
    
    # Per-edit-distance slice metrics
    for entry in sweep:
        d = entry["edit_distance"]
        key = f"attn_distance_edit_{d}"
        metrics[key] = float(entry["attn_distance_mean"])
    
    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Detect clearly broken runs that should skip the jury.
    
    Returns True if:
    - Any metric is NaN or inf
    - Headline correlation is not meaningfully above baseline (<= baseline + 0.05)
    - Correlation is negative (wrong direction)
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    
    headline = metrics.get("edit_distance_correlation")
    baseline = metrics.get("linear_baseline_correlation")
    
    if isinstance(headline, (int, float)) and isinstance(baseline, (int, float)):
        # If headline is worse than or barely better than baseline, it's broken
        if headline <= baseline + 0.05:
            return True
        # Negative correlation means attention distance decreases with edit distance
        if headline < 0:
            return True
    
    return False