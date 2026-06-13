import numpy as np
import math

VERSION = 1


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of y over x. NumPy 2.0 removed np.trapz, so we
    implement it directly. Returns 0.0 for fewer than two points."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))


def score(payload: dict) -> dict[str, float | int]:
    """Compute attention span metrics from payload."""
    # Validate required keys
    required_keys = [
        "version", "canonical_seq_len", "canonical_distances",
        "samples_per_distance", "sweep", "attention_span_auc"
    ]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k}")

    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version: {payload['version']}")

    sweep = payload["sweep"]
    if not sweep:
        raise ValueError("Empty sweep")

    # Extract distances and attention values
    distances = np.array([s["distance"] for s in sweep], dtype=np.float64)
    mean_attns = np.array([s["mean_attention_on_target"] for s in sweep], dtype=np.float64)

    metrics: dict[str, float | int] = {"version": 1}

    # Per-slice metrics
    for s in sweep:
        d = s["distance"]
        key = f"attention_on_target_dist_{d}"
        metrics[key] = s["mean_attention_on_target"]

    # Headline AUC (recompute for consistency)
    log_distances = np.log2(distances)
    denom = _trapz(np.ones_like(mean_attns), log_distances)
    auc = _trapz(mean_attns, log_distances) / denom if denom > 0 else 0.0
    metrics["attention_span_auc_canonical"] = float(auc)

    # Robustness: ratio of attention at max distance to min distance
    if mean_attns[0] > 0:
        metrics["attention_span_robustness"] = float(mean_attns[-1] / mean_attns[0])
    else:
        metrics["attention_span_robustness"] = 0.0

    # Linear (uniform) baseline
    seq_len = payload["canonical_seq_len"]
    uniform_attn = 1.0 / seq_len
    metrics["linear_baseline_attention_span_auc"] = uniform_attn
    metrics["linear_baseline_attention_on_target_dist_1"] = uniform_attn

    # Lift over baseline
    metrics["lift_over_baseline_auc"] = float(auc - uniform_attn)

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """Return True if metrics indicate a clearly degenerate run."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    # AUC should beat uniform baseline by a meaningful margin
    auc = metrics.get("attention_span_auc_canonical")
    baseline = metrics.get("linear_baseline_attention_span_auc")
    if isinstance(auc, (int, float)) and isinstance(baseline, (int, float)):
        if auc <= baseline * 1.01:  # allow tiny numerical noise
            return True
    return False

GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU
