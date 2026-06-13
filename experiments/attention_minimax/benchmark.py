import math
from typing import Any

VERSION = 1

# Minimum GPU slots for attempts (pipeline clamps to >= 1)
GPU_REQUIREMENT = 1


def score(payload: dict[str, Any]) -> dict[str, float | int]:
    """
    Compute metrics from payload. Pure Python, deterministic, side-effect free.
    """
    # --- Contract validation ---
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported payload version: {payload.get('version')}")
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    if "sweep_alphas" not in payload:
        raise KeyError("payload missing 'sweep_alphas'")

    sweep = payload["sweep"]
    expected_alphas = payload["sweep_alphas"]
    if len(sweep) != len(expected_alphas):
        raise ValueError(f"sweep length {len(sweep)} != sweep_alphas length {len(expected_alphas)}")

    # Build lookup by alpha (rounded to 1 decimal for safety)
    by_alpha = {}
    for record in sweep:
        if not isinstance(record, dict):
            raise ValueError("each sweep record must be a dict")
        alpha = record.get("alpha")
        if alpha is None:
            raise KeyError("sweep record missing 'alpha'")
        if "max_weight" not in record:
            raise KeyError(f"sweep record alpha={alpha} missing 'max_weight'")
        if "entropy" not in record:
            raise KeyError(f"sweep record alpha={alpha} missing 'entropy'")
        if "uniform_kl" not in record:
            raise KeyError(f"sweep record alpha={alpha} missing 'uniform_kl'")
        by_alpha[round(alpha, 1)] = record

    # Verify all expected alphas present
    for a in expected_alphas:
        if round(a, 1) not in by_alpha:
            raise KeyError(f"missing sweep record for alpha={a}")

    # --- Linear attention baseline (same sweep, deterministic) ---
    # We recompute it here so the baseline is always consistent.
    # Embeddings from task.py (replicated for independence)
    _EMBEDDING_SEED = 42
    _D_MODEL = 32
    _TOKEN_TYPES = ["TARGET", "DISTRACTOR_A", "DISTRACTOR_B", "DISTRACTOR_C"]
    _rng_embed = __import__('numpy').random.default_rng(_EMBEDDING_SEED)
    _TOKEN_EMBEDDINGS = {
        tok: _rng_embed.normal(size=_D_MODEL).astype(np.float32)
        for tok in _TOKEN_TYPES
    }
    # MUST match task.py exactly: orthogonalize against TARGET only.
    _NOISE_VEC = _rng_embed.normal(size=_D_MODEL).astype(np.float32)
    _e_target = _TOKEN_EMBEDDINGS["TARGET"]
    _NOISE_VEC -= _e_target * (np.dot(_NOISE_VEC, _e_target) / np.dot(_e_target, _e_target))
    _NOISE_VEC = _NOISE_VEC / np.linalg.norm(_NOISE_VEC)
    _DISTRACTOR_EMBEDS = np.stack([
        _TOKEN_EMBEDDINGS["DISTRACTOR_A"],
        _TOKEN_EMBEDDINGS["DISTRACTOR_B"],
        _TOKEN_EMBEDDINGS["DISTRACTOR_C"],
    ], axis=0)

    def _linear_attn(query, keys):
        scores = keys @ query
        scores = scores - np.min(scores) + 1e-8
        return scores / np.sum(scores)

    linear_regrets = {}
    for alpha in expected_alphas:
        query_vec = alpha * _TOKEN_EMBEDDINGS["TARGET"] + (1 - alpha) * _NOISE_VEC
        linear_weights = _linear_attn(query_vec, _DISTRACTOR_EMBEDS)
        linear_max = float(np.max(linear_weights))
        linear_regrets[round(alpha, 1)] = linear_max - 1.0/3.0

    # --- Metrics computation ---
    metrics: dict[str, float | int] = {"version": VERSION}

    # Headline: minimax regret at canonical condition (alpha=0)
    if 0.0 not in by_alpha:
        raise KeyError("canonical condition alpha=0.0 missing from sweep")
    canonical_record = by_alpha[0.0]
    canonical_max_weight = canonical_record["max_weight"]
    minimax_regret_canonical = canonical_max_weight - 1.0/3.0
    metrics["minimax_regret_canonical"] = float(minimax_regret_canonical)

    # Per-slice regrets
    for alpha in expected_alphas:
        rec = by_alpha[round(alpha, 1)]
        regret = rec["max_weight"] - 1.0/3.0
        key = f"minimax_regret_alpha_{alpha:.1f}".replace(".", "p")
        metrics[key] = float(regret)

    # Per-slice entropy and uniform_kl
    for alpha in expected_alphas:
        rec = by_alpha[round(alpha, 1)]
        key_ent = f"entropy_alpha_{alpha:.1f}".replace(".", "p")
        key_kl = f"uniform_kl_alpha_{alpha:.1f}".replace(".", "p")
        metrics[key_ent] = float(rec["entropy"])
        metrics[key_kl] = float(rec["uniform_kl"])

    # Linear baseline metrics
    linear_canonical = linear_regrets[0.0]
    metrics["linear_baseline_regret_canonical"] = float(linear_canonical)
    metrics["lift_over_linear_canonical"] = float(linear_canonical - minimax_regret_canonical)

    return metrics


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """
    Pipeline hook: return True if metrics indicate a clearly broken attempt.
    Never returns True for a borderline-but-real result.
    """
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Regret worse than linear baseline at canonical = mechanism hurts
    regret = metrics.get("minimax_regret_canonical")
    baseline = metrics.get("linear_baseline_regret_canonical")
    if isinstance(regret, (int, float)) and isinstance(baseline, (int, float)):
        # If regret > baseline * 2, something is very wrong (e.g., attention collapsed to one distractor)
        if regret > baseline * 2.0:
            return True

    # Negative entropy or KL = math error
    for k, v in metrics.items():
        if k.startswith("entropy_") or k.startswith("uniform_kl_"):
            if isinstance(v, (int, float)) and v < -1e-6:
                return True

    return False


# Import numpy at module level for baseline computation
import numpy as np