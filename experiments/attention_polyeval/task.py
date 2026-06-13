import numpy as np
from dataclasses import dataclass
from typing import Callable

ModelFn = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class Batch:
    inputs: np.ndarray          # [seq_len, d_model]
    targets: dict[int, np.ndarray]  # degree -> [seq_len, d_model]
    config: dict

def generate(seed: int = 0) -> Batch:
    """Generate deterministic synthetic data for polynomial evaluation."""
    rng = np.random.default_rng(seed)
    
    # Canonical configuration
    seq_len = 128
    d_model = 64
    n_heads = 4
    d_head = 16
    input_scale = 1.0
    degrees = [1, 2, 3]
    
    # Generate inputs: [seq_len, d_model] ~ Uniform[-scale, scale]
    inputs = rng.uniform(-input_scale, input_scale, size=(seq_len, d_model)).astype(np.float32)
    
    # Compute target polynomials elementwise
    targets = {}
    for deg in degrees:
        targets[deg] = np.power(inputs, deg).astype(np.float32)
    
    config = {
        "seed": seed,
        "input_scale": input_scale,
        "degrees": degrees,
        "seq_len": seq_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "d_head": d_head,
    }
    
    return Batch(inputs=inputs, targets=targets, config=config)


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn on generated batch, compute metrics vs polynomial targets."""
    batch = generate(seed=42)  # Canonical seed
    
    # Run the model
    outputs = model_fn(batch.inputs)  # [seq_len, d_model]
    
    if outputs.shape != batch.inputs.shape:
        raise ValueError(f"model_fn output shape {outputs.shape} != input shape {batch.inputs.shape}")
    
    # Compute metrics per degree
    sweep = []
    linear_baseline = []
    
    for deg in batch.config["degrees"]:
        target = batch.targets[deg]  # [seq_len, d_model]
        
        # Model metrics
        mse = float(np.mean((outputs - target) ** 2))
        corr = float(_pearson_correlation(outputs.flatten(), target.flatten()))
        var_target = float(np.var(target))
        r2 = 1.0 - mse / var_target if var_target > 0 else 0.0
        
        sweep.append({
            "degree": deg,
            "mse": mse,
            "correlation": corr,
            "variance_explained": r2,
        })
        
        # Linear baseline: best *affine* predictor a*x + b fit globally.
        # This is the strongest a linear map can do; the intercept matters.
        # For x ~ Uniform[-1, 1], E[x] = 0, Var(x) = 1/3.
        # Cov(x, x^d) = E[x^{d+1}] = 0 for even d, non-zero for odd d, so for
        # even degrees the best fit collapses to the constant mean (R^2 = 0),
        # which is exactly the "a linear map cannot capture this" reference.
        # Without the intercept the baseline would predict ~0 and score a
        # spuriously negative R^2, inflating the non-linear lift.
        x_flat = batch.inputs.flatten()
        y_flat = target.flatten()
        var_x = float(np.var(x_flat))
        if var_x > 0:
            cov = float(np.cov(x_flat, y_flat)[0, 1])
            slope = cov / var_x
        else:
            slope = 0.0
        intercept = float(np.mean(y_flat) - slope * np.mean(x_flat))
        baseline_pred = batch.inputs * slope + intercept
        
        mse_lin = float(np.mean((baseline_pred - target) ** 2))
        corr_lin = float(_pearson_correlation(baseline_pred.flatten(), target.flatten()))
        r2_lin = 1.0 - mse_lin / var_target if var_target > 0 else 0.0
        
        linear_baseline.append({
            "degree": deg,
            "mse": mse_lin,
            "correlation": corr_lin,
            "variance_explained": r2_lin,
        })
    
    return {
        "version": 1,
        "config": batch.config,
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


def random_model_fn() -> ModelFn:
    """Return a dummy model_fn that returns zeros of the correct shape."""
    def _fn(inputs: np.ndarray) -> np.ndarray:
        return np.zeros_like(inputs, dtype=np.float32)
    return _fn


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Pearson correlation, handling edge cases."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_centered = x - x_mean
    y_centered = y - y_mean
    numerator = np.sum(x_centered * y_centered)
    denom_x = np.sum(x_centered ** 2)
    denom_y = np.sum(y_centered ** 2)
    if denom_x == 0 or denom_y == 0:
        return 0.0
    return float(numerator / np.sqrt(denom_x * denom_y))