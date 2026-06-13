import numpy as np
from dataclasses import dataclass
from typing import Callable, Any

from benchmark import VERSION

@dataclass(frozen=True)
class Batch:
    x1: np.ndarray      # (B, 1)
    x2: np.ndarray      # (B, 1)
    alpha: np.ndarray   # (B, 1)
    beta: np.ndarray    # (B, 1)
    target_positions: int = 5  # positions 3-7

def generate(seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    B = 256
    x1 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
    x2 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
    # Canonical coefficients: α=1, β=1
    alpha = np.ones((B, 1), dtype=np.float32)
    beta = np.ones((B, 1), dtype=np.float32)
    return Batch(x1=x1, x2=x2, alpha=alpha, beta=beta)

def _sweep_coeffs():
    """16 pairs: α,β ∈ {0, ±1, ±2} excluding (0,0)"""
    vals = [0.0, 1.0, -1.0, 2.0, -2.0]
    pairs = [(a, b) for a in vals for b in vals if not (a == 0 and b == 0)]
    return pairs

def evaluate(model_fn: Callable[[Batch], np.ndarray]) -> dict:
    # Canonical evaluation
    batch = generate(seed=42)  # fixed seed for canonical
    pred = model_fn(batch)  # (B, 5)
    if pred.ndim != 2 or pred.shape[1] != 5:
        raise ValueError(f"model_fn must return (B, 5), got {pred.shape}")
    target = batch.alpha * batch.x1 + batch.beta * batch.x2  # (B, 1)
    target = np.repeat(target, 5, axis=1)  # (B, 5)
    mse_canon = float(np.mean((pred - target) ** 2))
    mae_canon = float(np.mean(np.abs(pred - target)))
    var_target = float(np.var(target))
    r2_canon = 1.0 - mse_canon / var_target if var_target > 0 else 0.0

    # Sweep evaluation
    sweep = []
    for alpha_val, beta_val in _sweep_coeffs():
        # Create batch with these coefficients
        rng = np.random.default_rng(123)  # same x1,x2 across sweep
        B = 256
        x1 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
        x2 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
        alpha_arr = np.full((B, 1), alpha_val, dtype=np.float32)
        beta_arr = np.full((B, 1), beta_val, dtype=np.float32)
        sweep_batch = Batch(x1=x1, x2=x2, alpha=alpha_arr, beta=beta_arr)
        pred_s = model_fn(sweep_batch)
        target_s = alpha_arr * x1 + beta_arr * x2
        target_s = np.repeat(target_s, 5, axis=1)
        mse = float(np.mean((pred_s - target_s) ** 2))
        mae = float(np.mean(np.abs(pred_s - target_s)))
        var_t = float(np.var(target_s))
        r2 = 1.0 - mse / var_t if var_t > 0 else 0.0
        sweep.append({
            "alpha": float(alpha_val),
            "beta": float(beta_val),
            "mse": mse,
            "mae": mae,
            "r2": r2,
        })

    # Baseline: mean predictor
    mean_target = float(np.mean(target))
    baseline_pred = np.full_like(pred, mean_target)
    baseline_mse = float(np.mean((baseline_pred - target) ** 2))
    baseline_r2 = 1.0 - baseline_mse / var_target if var_target > 0 else 0.0

    return {
        "version": VERSION,
        "canonical": {
            "pred": pred.tolist(),
            "target": target.tolist(),
        },
        "sweep": sweep,
        "config": {
            "seq_len": 8,
            "batch_size": 256,
            "d_model": 32,
            "d_head": 32,
            "num_target_positions": 5,
        },
        "baseline": {
            "mse_canonical": baseline_mse,
            "r2_canonical": baseline_r2,
        }
    }

def random_model_fn() -> Callable[[Batch], np.ndarray]:
    def _fn(batch: Batch) -> np.ndarray:
        B = batch.x1.shape[0]
        return np.zeros((B, 5), dtype=np.float32)
    return _fn