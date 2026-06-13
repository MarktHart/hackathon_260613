import math
from typing import Any
import numpy as np


VERSION = 1


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute AUROC for binary labels. scores: higher = more positive.
    Returns 0.5 for degenerate cases (all same label or all same score).
    """
    if len(scores) != len(labels):
        raise ValueError("scores and labels length mismatch")
    if len(scores) < 2:
        return 0.5

    # Check for degenerate labels
    if np.all(labels == labels[0]):
        return 0.5

    # Rank scores (average rank for ties)
    order = np.argsort(scores)
    ranked_scores = scores[order]
    ranked_labels = labels[order]

    # Compute AUC via Mann-Whitney U
    pos_scores = ranked_scores[ranked_labels == 1]
    neg_scores = ranked_scores[ranked_labels == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5

    # Count concordant pairs
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    concordant = 0
    for ps in pos_scores:
        concordant += np.sum(neg_scores < ps)
        concordant += 0.5 * np.sum(neg_scores == ps)

    auc = concordant / (n_pos * n_neg)
    return float(auc)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returns 0.0 if either is constant or length < 2."""
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def score(payload: dict) -> dict[str, float | int]:
    """
    Compute metrics from payload. Raises KeyError/ValueError on contract violation.
    """
    # ---- Validate payload ----
    required_keys = ('version', 'sweep', 'config')
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing payload key: {k}")

    if payload['version'] != VERSION:
        raise ValueError(f"Payload version {payload['version']} != benchmark VERSION {VERSION}")

    sweep = payload['sweep']
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Payload 'sweep' must be a non-empty list")

    # ---- Extract per-scale data ----
    scales = []
    sat_scores = []
    mean_entropies = []
    max_weights = []
    ref_mean_entropies = []
    ref_max_weights = []

    for rec in sweep:
        for k in ('logit_scale', 'saturation_score', 'mean_entropy', 'max_attn_weight',
                  'ref_mean_entropy', 'ref_max_attn_weight'):
            if k not in rec:
                raise KeyError(f"Sweep record missing key: {k}")

        scales.append(float(rec['logit_scale']))
        sat_scores.append(float(rec['saturation_score']))
        mean_entropies.append(float(rec['mean_entropy']))
        max_weights.append(float(rec['max_attn_weight']))
        ref_mean_entropies.append(float(rec['ref_mean_entropy']))
        ref_max_weights.append(float(rec['ref_max_attn_weight']))

    scales = np.array(scales)
    sat_scores = np.array(sat_scores)
    mean_entropies = np.array(mean_entropies)
    max_weights = np.array(max_weights)
    ref_mean_entropies = np.array(ref_mean_entropies)
    ref_max_weights = np.array(ref_max_weights)

    # ---- Ground truth saturation labels ----
    # Saturated iff logit_scale >= 10.0 (canonical threshold)
    saturated_labels = (scales >= 10.0).astype(int)

    # ---- Headline: AUROC for saturation detection ----
    saturation_detection_auroc = _roc_auc(sat_scores, saturated_labels)

    # ---- Baselines ----
    # Linear baseline: saturation_score = logit_scale (monotonic with scale)
    linear_baseline_auroc = _roc_auc(scales, saturated_labels)
    # Entropy oracle: saturation_score = -mean_entropy (entropy drops in saturation)
    entropy_baseline_auroc = _roc_auc(-ref_mean_entropies, saturated_labels)

    # ---- Per-slice metrics ----
    metrics: dict[str, float | int] = {
        'version': VERSION,
        'saturation_detection_auroc': saturation_detection_auroc,
        'linear_baseline_auroc': linear_baseline_auroc,
        'entropy_baseline_auroc': entropy_baseline_auroc,
        'lift_over_linear': saturation_detection_auroc - linear_baseline_auroc,
        'lift_over_entropy': entropy_baseline_auroc - saturation_detection_auroc,
    }

    # Per-scale values
    for i, s in enumerate(scales):
        scale_str = f"{s:.1f}".replace('.', 'p').replace('-', 'm')
        metrics[f'mean_entropy_logit_{scale_str}'] = mean_entropies[i]
        metrics[f'max_weight_logit_{scale_str}'] = max_weights[i]
        metrics[f'saturation_score_logit_{scale_str}'] = sat_scores[i]
        metrics[f'ref_mean_entropy_logit_{scale_str}'] = ref_mean_entropies[i]
        metrics[f'ref_max_weight_logit_{scale_str}'] = ref_max_weights[i]

    # Sweep-level entropy correlation: how well does the attempt's mean entropy
    # track the analytic reference across the scale sweep (Pearson r in [-1, 1]).
    if len(scales) >= 2:
        ent_corr = _pearson_r(mean_entropies, ref_mean_entropies)
        metrics['entropy_correlation_sweep'] = ent_corr

    return metrics


def is_obviously_broken(metrics: dict) -> bool:
    """
    Pipeline hook: return True if metrics indicate a clearly failed attempt.
    """
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # The saturation label is `logit_scale >= 10`, so scale itself is a perfect
    # separator and `linear_baseline_auroc` is always ~1.0. We therefore CANNOT
    # require the attempt to beat that (oracle) baseline — nothing can. Instead
    # flag attempts whose detector is no better than random (AUROC <= 0.5).
    auroc = metrics.get('saturation_detection_auroc')
    if isinstance(auroc, (int, float)) and not isinstance(auroc, bool):
        if auroc <= 0.5:
            return True

    return False


# Optional: GPU requirement for attempts (default 1)
GPU_REQUIREMENT = 1