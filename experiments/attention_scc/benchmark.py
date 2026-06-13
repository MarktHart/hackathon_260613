import math
from typing import Any, Dict, List, Tuple

VERSION = 1

GPU_REQUIREMENT = 1


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _fmt_rho(rho: float) -> str:
    """Format a rho value for a metric key: 0.25 -> '0p25', 1.0 -> '1p0'."""
    return str(float(rho)).replace(".", "p").replace("-", "neg")


def _trapz_norm(xs: List[float], ys: List[float]) -> float:
    """Trapezoidal integral of ys vs xs, normalised by (max_x - min_x).

    Returns a value on the same scale as ys (an average over the rho range).
    With ys in [0, 1] the result is in [0, 1]. Returns ys[0] for a single
    point and 0.0 for a degenerate (zero-width) range.
    """
    n = len(xs)
    if n == 0:
        return 0.0
    if n == 1:
        return float(ys[0])
    area = 0.0
    for i in range(n - 1):
        area += 0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i])
    width = xs[-1] - xs[0]
    if width <= 0:
        return float(ys[0])
    return float(area / width)


def _capacity(xs: List[float], ys: List[float], thr: float) -> float:
    """Largest rho at which target attention >= thr (linear interpolation).

    xs are assumed sorted ascending. If ys never reaches thr, returns min(xs);
    if always above, returns max(xs).
    """
    if not xs:
        raise ValueError("capacity: empty sweep")
    cap = None
    for i in range(len(xs)):
        if ys[i] >= thr:
            cap = xs[i]
            # If the next point drops below, interpolate the crossing.
            if i + 1 < len(xs) and ys[i + 1] < thr:
                denom = ys[i] - ys[i + 1]
                if denom > 0:
                    frac = (ys[i] - thr) / denom
                    cap = xs[i] + frac * (xs[i + 1] - xs[i])
    if cap is None:
        cap = xs[0]
    return float(cap)


def _sorted_sweep(sweep: List[Dict[str, Any]]) -> Tuple[List[float], List[float], List[float]]:
    """Validate and return (rhos, target_means, chance_levels) sorted by rho."""
    rows = []
    for rec in sweep:
        for key in ("rho", "target_attention_mean", "chance_level"):
            if key not in rec:
                raise KeyError(f"sweep record missing required key '{key}': {rec}")
        rho = float(rec["rho"])
        mean = float(rec["target_attention_mean"])
        chance = float(rec["chance_level"])
        if not (math.isfinite(rho) and math.isfinite(mean) and math.isfinite(chance)):
            raise ValueError(f"non-finite value in sweep record: {rec}")
        rows.append((rho, mean, chance))
    rows.sort(key=lambda r: r[0])
    rhos = [r[0] for r in rows]
    means = [r[1] for r in rows]
    chances = [r[2] for r in rows]
    return rhos, means, chances


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def score(payload: Dict[str, Any]) -> Dict[str, float | int]:
    """Compute the flat metric dict from a task.evaluate payload."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    if "version" not in payload:
        raise KeyError("payload missing 'version'")
    if payload["version"] != VERSION:
        raise ValueError(
            f"unsupported payload version {payload['version']!r}; expected {VERSION}"
        )
    if "sweep" not in payload:
        raise KeyError("payload missing 'sweep'")
    sweep = payload["sweep"]
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("payload 'sweep' must be a non-empty list")

    rhos, means, chances = _sorted_sweep(sweep)

    metrics: Dict[str, float | int] = {"version": VERSION}

    # Per-slice values: method and baseline at each rho.
    for rho, mean, chance in zip(rhos, means, chances):
        key = _fmt_rho(rho)
        metrics[f"scc_rho_{key}"] = mean
        metrics[f"linear_baseline_rho_{key}"] = chance

    # Headline: normalised area under the capacity curve.
    scc_auc = _trapz_norm(rhos, means)
    baseline_auc = _trapz_norm(rhos, chances)
    metrics["scc_auc"] = scc_auc
    metrics["linear_baseline_scc_auc"] = baseline_auc
    metrics["lift_over_linear_auc"] = scc_auc - baseline_auc

    # Canonical slice (rho = 1.0, the critical/complete ratio) if present.
    for rho, mean in zip(rhos, means):
        if abs(rho - 1.0) < 1e-9:
            metrics["scc_auc_canonical"] = mean
            break

    # Capacity thresholds: highest rho sustaining target attention.
    metrics["capacity_rho_0p5"] = _capacity(rhos, means, 0.5)
    metrics["capacity_rho_0p9"] = _capacity(rhos, means, 0.9)

    return metrics


def is_obviously_broken(metrics: Dict[str, float | int]) -> bool:
    """Mechanical degeneracy check; True short-circuits the jury."""
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True

    scc_auc = metrics.get("scc_auc")
    baseline = metrics.get("linear_baseline_scc_auc")
    if isinstance(scc_auc, (int, float)) and isinstance(baseline, (int, float)):
        # No meaningful lift over uniform attention -> nothing learned.
        if scc_auc <= baseline + 1e-6:
            return True

    return False
