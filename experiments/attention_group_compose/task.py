import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    """Container for the synthetic evaluation set."""
    # List of (attn_a, attn_b, true_composition, noise_level) tuples
    queries: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]]
    n: int
    noise_levels: list[float]
    num_pairs_per_level: int
    seed: int

def _make_permutation_matrix(n: int, k: int) -> np.ndarray:
    """Permutation matrix for rotation by k in C_n."""
    P = np.zeros((n, n))
    for i in range(n):
        P[i, (i + k) % n] = 1.0
    return P

def _noisy_permutation(n: int, k: int, noise: float, rng: np.random.Generator) -> np.ndarray:
    """Softmax(noisy logits) where clean logits are log(P) = large on 1s, -inf on 0s.
    We approximate log(P) with large finite values: +L for 1, -L for 0."""
    L = 20.0  # large logit ≈ hard permutation
    logits = np.full((n, n), -L)
    for i in range(n):
        logits[i, (i + k) % n] = L
    if noise > 0:
        logits += noise * rng.standard_normal((n, n))
    # Softmax per row
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

def generate(seed: int = 0) -> Batch:
    """Deterministic generation of composition queries."""
    n = 6
    # Noise levels are in *logit units* (added to logits of magnitude L=20 before
    # softmax). They are scaled so the sweep spans clean -> chance: sigma=0 is an
    # exact hard permutation, sigma=20 (canonical) meaningfully corrupts the
    # matrices while leaving headroom for a real method, sigma=40 approaches chance.
    noise_levels = [0.0, 10.0, 20.0, 30.0, 40.0]
    num_pairs = 200
    
    rng = np.random.default_rng(seed)
    queries = []
    
    for noise in noise_levels:
        for _ in range(num_pairs):
            # Sample two random group elements (rotations)
            k1 = rng.integers(0, n)
            k2 = rng.integers(0, n)
            
            # Noisy attention matrices
            A = _noisy_permutation(n, k1, noise, rng)
            B = _noisy_permutation(n, k2, noise, rng)
            
            # True composition: rotation by (k1 + k2) mod n, clean permutation
            true_k = (k1 + k2) % n
            true_composition = _make_permutation_matrix(n, true_k)
            
            queries.append((A, B, true_composition, noise))
    
    return Batch(
        queries=queries,
        n=n,
        noise_levels=noise_levels,
        num_pairs_per_level=num_pairs,
        seed=seed
    )

# ModelFn signature: (attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn on all queries, aggregate errors per noise level."""
    batch = generate(seed=0)  # Fixed seed for canonical condition
    
    # Accumulate errors per noise level
    errors_by_noise: dict[float, list[float]] = {nl: [] for nl in batch.noise_levels}
    baseline_errors_by_noise: dict[float, list[float]] = {nl: [] for nl in batch.noise_levels}
    
    for attn_a, attn_b, true_comp, noise in batch.queries:
        # Attempt's prediction
        pred = model_fn(attn_a, attn_b)
        
        # Normalized Frobenius error
        err = np.linalg.norm(pred - true_comp, 'fro') / batch.n
        errors_by_noise[noise].append(err)
        
        # Linear baseline: naive matrix multiply
        baseline_pred = attn_a @ attn_b
        baseline_err = np.linalg.norm(baseline_pred - true_comp, 'fro') / batch.n
        baseline_errors_by_noise[noise].append(baseline_err)
    
    # Build sweep records
    sweep = []
    for noise in batch.noise_levels:
        sweep.append({
            "noise_level": noise,
            "frobenius_error": float(np.mean(errors_by_noise[noise])),
            "linear_baseline_error": float(np.mean(baseline_errors_by_noise[noise])),
            "num_pairs": batch.num_pairs_per_level
        })
    
    return {
        "version": 1,
        "config": {
            "group": "cyclic",
            "n": batch.n,
            "noise_levels": batch.noise_levels,
            "num_pairs_per_level": batch.num_pairs_per_level,
            "seed": batch.seed
        },
        "sweep": sweep
    }

def random_model_fn() -> ModelFn:
    """Returns a model_fn that outputs uniform stochastic matrices (max entropy)."""
    def _random_fn(attn_a: np.ndarray, attn_b: np.ndarray) -> np.ndarray:
        n = attn_a.shape[0]
        # Uniform row-stochastic matrix
        return np.full((n, n), 1.0 / n)
    return _random_fn