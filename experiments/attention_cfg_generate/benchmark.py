import math
from typing import Dict, Any

VERSION = 1
GPU_REQUIREMENT = 1

SWEEP_DEPTHS = [1, 2, 3, 4, 5]


def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    """Compute metrics from the evaluation payload.

    See the goal README ("Metrics") for formulas and direction-of-better.
    """
    required_keys = ["version", "canonical_depth", "sweep"]
    for k in required_keys:
        if k not in payload:
            raise KeyError(f"Missing required payload key: {k}")

    if payload["version"] != VERSION:
        raise ValueError(f"Unsupported payload version: {payload['version']}")

    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("sweep must be a non-empty list of depth records")

    # Index records by depth, applying the zero-pairs edge case.
    attn_by_depth: Dict[int, float] = {}
    unif_by_depth: Dict[int, float] = {}
    pairs_by_depth: Dict[int, int] = {}
    for rec in sweep:
        if "depth" not in rec:
            raise KeyError("each sweep record must have a 'depth' key")
        d = int(rec["depth"])
        n_pairs = int(rec.get("n_pairs", 0))
        if n_pairs > 0:
            attn = float(rec.get("mean_attn_to_match", 0.0))
            unif = float(rec.get("mean_attn_uniform", 0.0))
        else:
            attn = 0.0
            unif = 0.0
        attn_by_depth[d] = attn
        unif_by_depth[d] = unif
        pairs_by_depth[d] = n_pairs

    out: Dict[str, float | int] = {"version": VERSION}

    # Per-slice metrics (default missing depths to 0.0).
    for d in SWEEP_DEPTHS:
        out[f"stack_attention_depth_{d}"] = attn_by_depth.get(d, 0.0)
        out[f"uniform_baseline_depth_{d}"] = unif_by_depth.get(d, 0.0)

    # Headline (canonical depth).
    cd = int(payload["canonical_depth"])
    canonical = attn_by_depth.get(cd, 0.0)
    canonical_unif = unif_by_depth.get(cd, 0.0)
    out["stack_attention_canonical"] = canonical
    out["lift_over_uniform_canonical"] = canonical - canonical_unif

    # Robustness: min/max over depths that actually have pairs.
    nonzero = [
        attn_by_depth[d]
        for d in SWEEP_DEPTHS
        if pairs_by_depth.get(d, 0) > 0
    ]
    if nonzero:
        mx = max(nonzero)
        out["stack_attention_robustness"] = (min(nonzero) / mx) if mx > 0 else 0.0
    else:
        out["stack_attention_robustness"] = 0.0

    return out


def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    """True if metrics indicate a clearly failed attempt (NaN/inf, or no lift
    over the chance baseline at the canonical depth)."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    canonical = metrics.get("stack_attention_canonical")
    baseline = metrics.get("uniform_baseline_depth_3")
    if isinstance(canonical, (int, float)) and isinstance(baseline, (int, float)):
        # A stack-tracking head should put well above chance mass on the match.
        if canonical <= baseline * 1.5:
            return True

    return False
