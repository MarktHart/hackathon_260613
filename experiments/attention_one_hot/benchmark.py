import math
from typing import Any

VERSION = 1

def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """
    Compute metrics from the payload returned by task.evaluate.

    Args:
        payload: Dict with keys exactly as specified in README.md payload contract.

    Returns:
        Flat dict of metrics. First key is 'version'.
    """
    # Validate payload structure
    required_keys = {"version", "canonical_length", "temperature", "d_model", "sweep"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise KeyError(f"payload missing keys: {missing}")

    if payload["version"] != 1:
        raise ValueError(f"Unsupported payload version: {payload['version']}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("sweep must be a non-empty list")

    required_record_keys = {"length", "target_pos", "attention_entropy", "peak_attention", "target_attention", "output_cosine"}
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] is not a dict")
        missing_rec = required_record_keys - set(rec.keys())
        if missing_rec:
            raise KeyError(f"sweep record {i} missing keys: {missing_rec}")

    metrics: dict[str, float | int] = {"version": VERSION}

    # Per-slice metrics
    canonical_L = payload["canonical_length"]
    for rec in sweep:
        L = rec["length"]
        L_key = f"length_{L}"

        target_attn = rec["target_attention"]
        peak_attn = rec["peak_attention"]
        entropy = rec["attention_entropy"]
        output_cos = rec["output_cosine"]

        # Main metric: target attention (bigger is better)
        metrics[f"one_hot_{L_key}"] = target_attn
        metrics[f"peak_attention_{L_key}"] = peak_attn
        metrics[f"entropy_{L_key}"] = entropy
        metrics[f"output_cosine_{L_key}"] = output_cos

        # Linear baseline: uniform attention = 1/L
        metrics[f"linear_baseline_one_hot_{L_key}"] = 1.0 / L

        # Headline: canonical length
        if L == canonical_L:
            metrics["one_hot_canonical"] = target_attn

    # Length robustness: min/max ratio of target_attention across sweep
    target_attentions = [rec["target_attention"] for rec in sweep]
    min_attn = min(target_attentions)
    max_attn = max(target_attentions)
    if max_attn > 0:
        metrics["length_robustness"] = min_attn / max_attn
    else:
        metrics["length_robustness"] = 0.0

    return metrics

def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """
    Detect clearly degenerate results without running the jury.
    Returns True if the attempt should be marked failed immediately.
    """
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Must beat the linear baseline at canonical length by a meaningful margin
    # Linear baseline at L=64 is 1/64 ≈ 0.0156. Require at least 2x baseline.
    canonical = metrics.get("one_hot_canonical")
    baseline = metrics.get("linear_baseline_one_hot_length_64")
    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        if canonical <= baseline * 2.0:
            return True

    # Entropy should be reasonable (not > log(L) which would be uniform)
    # At canonical length, max entropy is log(64) ≈ 4.16
    entropy = metrics.get("entropy_length_64")
    if isinstance(entropy, float) and entropy > 5.0:
        return True

    return False

# GPU requirement: this is a tiny synthetic task, 1 GPU is plenty
GPU_REQUIREMENT = 1