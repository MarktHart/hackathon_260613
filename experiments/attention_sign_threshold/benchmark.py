import math
from typing import Dict, List, Any

VERSION = 1


def _logistic_fit(cosines: List[float], means: List[float]) -> tuple[float, float]:
    """
    Fit a logistic function σ(α * (cos - θ)) to the data using a simple
    moment-based approximation, then refine with a few Newton steps.
    Returns (alpha, theta) where alpha = sharpness, theta = threshold (cos at 0.5).
    """
    n = len(cosines)
    if n < 3:
        return 0.0, 0.0

    # Initial guess: theta at median cosine where mean crosses 0.5
    # Scale means to [0,1] range
    mn, mx = min(means), max(means)
    if mx - mn < 1e-9:
        return 0.0, 0.0
    scaled = [(m - mn) / (mx - mn) for m in means]

    # Rough theta: interpolate where scaled crosses 0.5
    theta = 0.0
    for i in range(n - 1):
        if (scaled[i] - 0.5) * (scaled[i + 1] - 0.5) <= 0:
            # Linear interpolation
            t = (0.5 - scaled[i]) / (scaled[i + 1] - scaled[i] + 1e-12)
            theta = cosines[i] + t * (cosines[i + 1] - cosines[i])
            break

    # Rough alpha: use slope at theta via finite diff
    slopes = []
    for i in range(1, n - 1):
        dc = cosines[i + 1] - cosines[i - 1]
        dm = scaled[i + 1] - scaled[i - 1]
        if abs(dc) > 1e-9:
            slopes.append(dm / dc)
    alpha_init = max(slopes) if slopes else 1.0
    alpha_init = max(alpha_init, 0.1)

    # Newton refinement on logistic likelihood (simplified: least squares on logits)
    alpha, theta_ = alpha_init, theta
    for _ in range(10):
        grad_a = 0.0
        grad_t = 0.0
        hess_aa = 0.0
        hess_at = 0.0
        hess_tt = 0.0
        for c, m in zip(cosines, scaled):
            z = alpha * (c - theta_)
            sigma = 1.0 / (1.0 + math.exp(-z))
            # Derivatives of sigma w.r.t alpha and theta
            ds_dz = sigma * (1 - sigma)
            ds_da = ds_dz * (c - theta_)
            ds_dt = ds_dz * (-alpha)
            err = sigma - m
            grad_a += err * ds_da
            grad_t += err * ds_dt
            hess_aa += ds_da * ds_da
            hess_at += ds_da * ds_dt
            hess_tt += ds_dt * ds_dt
        # Solve 2x2
        det = hess_aa * hess_tt - hess_at * hess_at
        if abs(det) < 1e-12:
            break
        da = -(hess_tt * grad_a - hess_at * grad_t) / det
        dt = -(-hess_at * grad_a + hess_aa * grad_t) / det
        alpha += da
        theta_ += dt
        if abs(da) < 1e-6 and abs(dt) < 1e-6:
            break

    alpha = max(alpha, 0.0)
    return alpha, theta_


def _linear_baseline(cosines: List[float]) -> List[float]:
    """No-mechanism reference: a head whose attention weight is a *linear ramp*
    in the positive half, ``attention = max(0, cos)``.

    This is the strawman a real sign detector must beat. It already lives in
    [0, 1] (cos ∈ [-1, 1]), so it is directly comparable to the per-pair
    sigmoid attention the evaluator produces — no extra normalisation needed.
    A ramp transitions across cos = 0 only gradually, so its logistic-fit
    sharpness is finite and modest; a true sign detector should fit a much
    steeper logistic and so score ``lift_over_linear_sharpness`` > 1.
    """
    return [max(0.0, float(c)) for c in cosines]


def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    """
    Compute all metrics from the payload.
    """
    # ---- validate payload ----
    if "version" not in payload:
        raise KeyError("payload missing 'version'")
    if payload["version"] != 1:
        raise ValueError(f"unsupported payload version {payload['version']}")
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload 'sweep' must be a non-empty list")
    for i, rec in enumerate(sweep):
        if not isinstance(rec, dict):
            raise TypeError(f"sweep[{i}] is not a dict")
        for key in ("cosine", "mean_attention", "std_attention"):
            if key not in rec:
                raise KeyError(f"sweep[{i}] missing key '{key}'")

    cosines = [rec["cosine"] for rec in sweep]
    means = [rec["mean_attention"] for rec in sweep]
    stds = [rec["std_attention"] for rec in sweep]
    n = len(cosines)

    # ---- headline: logistic fit sharpness & threshold ----
    alpha, theta = _logistic_fit(cosines, means)

    # ---- per-slice finite-difference sharpness ----
    per_slice = {}
    for i in range(1, n - 1):
        c0, c1, c2 = cosines[i - 1], cosines[i], cosines[i + 1]
        m0, m1, m2 = means[i - 1], means[i], means[i + 1]
        dc = c2 - c0
        if abs(dc) > 1e-9:
            slope = (m2 - m0) / dc
        else:
            slope = 0.0
        # Format cosine for key: 0.0 -> 0p0, -0.5 -> m0p5
        c_str = f"{c1:.1f}".replace(".", "p").replace("-", "m")
        per_slice[f"sign_sharpness_cos_{c_str}"] = float(slope)

    # ---- linear baseline (flat under per-bin softmax) ----
    baseline_means = _linear_baseline(cosines)
    baseline_alpha, _ = _logistic_fit(cosines, baseline_means)
    # baseline_alpha should be ~0

    # ---- lift ----
    eps = 1e-9
    lift = alpha / max(baseline_alpha, eps) if baseline_alpha > eps else float('inf')
    if not math.isfinite(lift):
        lift = 1e9  # cap for JSON serialisation

    # ---- assemble metrics ----
    metrics = {
        "version": VERSION,
        "sign_sharpness_canonical": float(alpha),
        "sign_threshold_canonical": float(theta),
        "linear_baseline_sharpness_canonical": float(baseline_alpha),
        "lift_over_linear_sharpness": float(lift),
    }
    metrics.update(per_slice)
    return metrics


def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    """
    Pipeline hook: return True if the attempt is mechanically broken.
    Called right after main.py writes benchmark.json.
    """
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    # Sharpness should be positive (or zero)
    sharp = metrics.get("sign_sharpness_canonical", 0.0)
    if sharp < -1e-6:
        return True

    # Threshold should be in [-1, 1]
    theta = metrics.get("sign_threshold_canonical", 0.0)
    if theta < -1.0 - 1e-6 or theta > 1.0 + 1e-6:
        return True

    # Lift should be >= 1 (or very large if baseline ≈ 0)
    lift = metrics.get("lift_over_linear_sharpness", 0.0)
    if lift < 0.5:  # allow some noise below 1
        return True

    return False


# Pipeline hook: this goal only needs 1 GPU (for model forward passes)
GPU_REQUIREMENT = 1  # GPU slots; attempts must run on the GPU