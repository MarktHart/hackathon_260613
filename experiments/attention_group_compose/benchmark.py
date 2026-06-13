import math

VERSION = 1

def score(payload: dict) -> dict[str, float | int]:
    """Compute metrics from task payload."""
    # Validate payload structure
    if payload.get("version") != VERSION:
        raise ValueError(f"Payload version {payload.get('version')} != benchmark VERSION {VERSION}")
    
    sweep = payload.get("sweep")
    if not isinstance(sweep, list) or len(sweep) == 0:
        raise ValueError("Payload must contain non-empty 'sweep' list")
    
    # Helper to find record by noise level
    def get_record(noise: float) -> dict:
        for rec in sweep:
            if abs(rec["noise_level"] - noise) < 1e-9:
                return rec
        raise KeyError(f"No sweep record for noise_level={noise}")
    
    # Noise levels in canonical order (logit units; see task.generate)
    noise_levels = [0.0, 10.0, 20.0, 30.0, 40.0]
    
    # Compute fidelities (1 - error, clipped to [0, 1])
    def fidelity(err: float) -> float:
        return max(0.0, min(1.0, 1.0 - err))
    
    metrics: dict[str, float | int] = {"version": VERSION}
    
    # Per-slice fidelities
    for nl in noise_levels:
        rec = get_record(nl)
        err = rec["frobenius_error"]
        base_err = rec["linear_baseline_error"]
        key = f"composition_fidelity_noise_{nl:.1f}".replace('.', 'p')
        base_key = f"linear_baseline_fidelity_noise_{nl:.1f}".replace('.', 'p')
        metrics[key] = fidelity(err)
        metrics[base_key] = fidelity(base_err)
    
    # Headline: canonical noise = 20.0
    metrics["composition_fidelity_canonical"] = metrics["composition_fidelity_noise_20p0"]
    
    # Lift over baseline at canonical
    metrics["lift_over_baseline_canonical"] = (
        metrics["composition_fidelity_canonical"] - metrics["linear_baseline_fidelity_noise_20p0"]
    )
    
    # Robustness: span-normalized AUC of the fidelity curve over the noise sweep
    # (trapezoid rule). Dividing by the noise span keeps it in [0, 1] regardless
    # of the sweep endpoints: 1 = perfect fidelity at every level, 0 = chance.
    fidelities = [metrics[f"composition_fidelity_noise_{nl:.1f}".replace('.', 'p')] for nl in noise_levels]
    span = noise_levels[-1] - noise_levels[0]
    if span <= 0:
        raise ValueError("noise sweep must span a positive range to compute robustness")
    auc = 0.0
    for i in range(len(noise_levels) - 1):
        auc += (fidelities[i] + fidelities[i+1]) * 0.5 * (noise_levels[i+1] - noise_levels[i])
    metrics["composition_robustness"] = auc / span
    
    return metrics

def is_obviously_broken(metrics: dict) -> bool:
    """Short-circuit jury for clearly degenerate attempts."""
    # NaN / inf check
    for v in metrics.values():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return True
    
    # Fidelity worse than baseline at canonical (should at least beat naive matmul)
    fid = metrics.get("composition_fidelity_canonical")
    base = metrics.get("linear_baseline_fidelity_noise_20p0")
    if isinstance(fid, (int, float)) and isinstance(base, (int, float)):
        if fid <= base:
            return True
    
    # Robustness out of bounds
    rob = metrics.get("composition_robustness")
    if isinstance(rob, float) and (rob < -0.1 or rob > 1.1):
        return True
    
    return False

# One GPU slot is sufficient for attempts (they only run model_fn on CPU arrays)
GPU_REQUIREMENT = 1