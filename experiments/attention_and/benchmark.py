import math
from typing import Any

VERSION = 2

# Cosine sweep values (must match task.py)
COS_SWEEP = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
CANONICAL_COS = 0.7

# Analytic linear baseline: an additive (no-AND) linear probe that reads BOTH
# features with weight c. Logits are reported in the same units as the model's,
# i.e. x·q / sqrt(d) (see README "Payload contract"), so the baseline carries
# the same 1/sqrt(d) factor. With x = alpha*v_A + beta*v_B + noise (noise mean 0):
#   both present:    (c + c) / sqrt(d) = 2c / sqrt(d)
#   only A present:  c / sqrt(d)
#   only B present:  c / sqrt(d)
#   neither:         0
# Sharpness = (2c - max(c, c)) / sqrt(d) = c / sqrt(d).
# A genuine AND must *beat* this additive floor (lift_over_linear > 0).
def _linear_baseline_sharpness(c: float, d: float) -> float:
    return c / math.sqrt(d)

def _fmt_cos(c: float) -> str:
    # 0.0 -> 0p0, 0.3 -> 0p3, 0.7 -> 0p7, 1.0 -> 1p0
    return f"{c:.1f}".replace(".", "p")

def score(payload: dict[str, Any]) -> dict[str, float | int]:
    # ---- Input validation ----
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if payload.get("version") != VERSION:
        raise ValueError(f"payload version {payload.get('version')} != benchmark VERSION {VERSION}")
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    d = payload.get("d")
    if not isinstance(d, (int, float)) or isinstance(d, bool) or d <= 0:
        raise ValueError(f"payload['d'] must be a positive number, got {d!r}")
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) != len(COS_SWEEP):
        raise ValueError(f"sweep must be a list of length {len(COS_SWEEP)}")
    # Verify sweep order and cos_sim values
    for i, (expected_c, rec) in enumerate(zip(COS_SWEEP, sweep)):
        if not isinstance(rec, dict):
            raise ValueError(f"sweep[{i}] must be a dict")
        got_c = rec.get("cos_sim")
        if got_c != expected_c:
            raise ValueError(f"sweep[{i}].cos_sim = {got_c}, expected {expected_c}")
        required_keys = {"logit_AA", "logit_AB", "logit_A0", "logit_B0", "logit_00_A", "logit_00_B"}
        missing = required_keys - rec.keys()
        if missing:
            raise KeyError(f"sweep[{i}] missing keys: {missing}")
        for k in required_keys:
            v = rec[k]
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                raise ValueError(f"sweep[{i}].{k} must be a finite float, got {v}")

    # ---- Compute per-slice sharpness ----
    sharpness_by_c: dict[float, float] = {}
    for rec in sweep:
        c = rec["cos_sim"]
        # Average the two heads (A and B) for stability
        sharp_A = rec["logit_AA"] - max(rec["logit_A0"], rec["logit_B0"])
        sharp_B = rec["logit_AB"] - max(rec["logit_A0"], rec["logit_B0"])
        sharpness_by_c[c] = (sharp_A + sharp_B) / 2.0

    # ---- Build metrics dict ----
    metrics: dict[str, float | int] = {"version": VERSION}

    # Per-slice values and baselines
    for c in COS_SWEEP:
        key_c = _fmt_cos(c)
        sharp = sharpness_by_c[c]
        lin_base = _linear_baseline_sharpness(c, d)
        metrics[f"and_sharpness_cos_{key_c}"] = sharp
        metrics[f"linear_baseline_sharpness_cos_{key_c}"] = lin_base
        metrics[f"lift_over_linear_cos_{key_c}"] = sharp - lin_base

    # Canonical convenience metrics
    canon_key = _fmt_cos(CANONICAL_COS)
    metrics["and_sharpness_canonical"] = metrics[f"and_sharpness_cos_{canon_key}"]
    metrics["linear_baseline_sharpness_canonical"] = metrics[f"linear_baseline_sharpness_cos_{canon_key}"]
    metrics["lift_over_linear_canonical"] = metrics[f"lift_over_linear_cos_{canon_key}"]

    # Headline: superposition_robustness = mean sharpness / sharpness at c=1.0
    # If sharpness at c=1.0 is <= 0, robustness is 0 (degenerate).
    sharp_at_1 = sharpness_by_c[1.0]
    if sharp_at_1 > 0:
        mean_sharp = sum(sharpness_by_c.values()) / len(sharpness_by_c)
        robustness = mean_sharp / sharp_at_1
        # Clamp to [0, 1] for interpretability (negative sharpness means worse than baseline)
        robustness = max(0.0, min(1.0, robustness))
    else:
        robustness = 0.0
    metrics["superposition_robustness"] = robustness

    return metrics


def is_obviously_broken(metrics: dict[str, float | int]) -> bool:
    """
    Pipeline hook: return True if the attempt is mechanically broken.
    Does NOT judge scientific merit — only catches NaN/inf, missing keys,
    or performance worse than the linear baseline at the canonical condition.
    """
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Must have the canonical sharpness and its linear baseline
    sharp = metrics.get("and_sharpness_canonical")
    baseline = metrics.get("linear_baseline_sharpness_canonical")
    if not isinstance(sharp, (int, float)) or not isinstance(baseline, (int, float)):
        return True

    # If the method doesn't beat the additive linear baseline by at least a
    # small margin at the canonical condition, it's not implementing an AND.
    # Both quantities are in x·q/sqrt(d) units; baseline at c=0.7 is
    # 0.7/sqrt(d). Threshold: 1.5x the baseline.
    if sharp <= baseline * 1.5:
        return True

    return False