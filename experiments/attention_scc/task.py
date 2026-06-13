import numpy as np
from dataclasses import dataclass
from typing import Protocol

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
class ModelFn(Protocol):
    def __call__(self, Q: np.ndarray, K: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Batch:
    """Pre-generated problem instances for the SCC (superposition capacity
    curve) sweep. For each rho: N_INSTANCES instances of (Q, K, target_idx).
    """
    d: int
    snr_db: float
    rhos: tuple[float, ...]
    # Per rho: list of (Q, K_mat, target_idx) tuples.
    instances: dict[float, list[tuple[np.ndarray, np.ndarray, int]]]


# ----------------------------------------------------------------------
# Constants (canonical condition)
# ----------------------------------------------------------------------
CANONICAL_D = 64
CANONICAL_SNR_DB = 10.0
CANONICAL_RHOS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_INSTANCES = 100


# ----------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------
def generate(seed: int = 0) -> Batch:
    """Deterministic generation of the SCC benchmark batch.

    Same seed -> identical batch. The query equals the target key plus
    isotropic Gaussian noise at a fixed SNR; the K keys are random unit
    vectors in R^d.
    """
    instances: dict[float, list[tuple[np.ndarray, np.ndarray, int]]] = {}

    for rho in CANONICAL_RHOS:
        K = max(1, int(round(rho * CANONICAL_D)))

        rho_instances = []
        for inst_idx in range(N_INSTANCES):
            # Per-instance seed, reproducible across rhos.
            inst_seed = int(seed) * 10_000 + int(round(rho * 1000)) * 100 + inst_idx
            inst_rng = np.random.default_rng(inst_seed)

            # K random unit key vectors in R^d.
            K_mat = inst_rng.normal(size=(K, CANONICAL_D)).astype(np.float32)
            K_mat = K_mat / np.linalg.norm(K_mat, axis=1, keepdims=True)

            target_idx = int(inst_rng.integers(0, K))
            target_key = K_mat[target_idx].copy()

            # Query = target_key + Gaussian noise at the specified SNR.
            # ||target_key||^2 = 1, so noise variance per dim = 1/(d * 10^(SNR/10)).
            noise_var = 1.0 / (CANONICAL_D * (10 ** (CANONICAL_SNR_DB / 10.0)))
            noise = inst_rng.normal(
                scale=float(np.sqrt(noise_var)), size=CANONICAL_D
            ).astype(np.float32)
            Q = target_key + noise
            Q = Q / np.linalg.norm(Q)

            rho_instances.append((Q, K_mat, target_idx))

        instances[rho] = rho_instances

    return Batch(
        d=CANONICAL_D,
        snr_db=CANONICAL_SNR_DB,
        rhos=CANONICAL_RHOS,
        instances=instances,
    )


# ----------------------------------------------------------------------
# Random model function (for smoke test)
# ----------------------------------------------------------------------
def random_model_fn() -> ModelFn:
    """Return a ModelFn that outputs a uniform distribution over the K keys.

    Pure NumPy, no torch, no GPU. Matches the ModelFn signature exactly
    (takes Q, K; returns a length-K probability vector).
    """
    def _uniform_attention(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
        K_len = K.shape[0]
        return np.ones(K_len, dtype=np.float32) / K_len

    return _uniform_attention


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over the canonical batch and return the score payload."""
    batch = generate(seed=0)  # Fixed seed for the canonical condition.

    sweep_records = []
    for rho in batch.rhos:
        instances = batch.instances[rho]
        K = len(instances[0][1])  # number of keys at this rho

        target_attentions = []
        for Q, K_mat, target_idx in instances:
            attn = np.asarray(model_fn(Q, K_mat), dtype=np.float64)
            if attn.shape != (K,):
                raise ValueError(
                    f"model_fn returned shape {attn.shape}, expected ({K},)"
                )
            if not np.isfinite(attn).all():
                raise ValueError("model_fn output contains non-finite values")
            if np.any(attn < -1e-6):
                raise ValueError("model_fn output contains negative values")
            if not np.isclose(np.sum(attn), 1.0, rtol=1e-4, atol=1e-4):
                raise ValueError(
                    f"model_fn output does not sum to 1: {float(np.sum(attn))}"
                )
            target_attentions.append(float(attn[target_idx]))

        target_attentions = np.asarray(target_attentions, dtype=np.float64)
        sweep_records.append({
            "rho": float(rho),
            "K": int(K),
            "target_attention_mean": float(np.mean(target_attentions)),
            "target_attention_std": float(np.std(target_attentions)),
            "chance_level": float(1.0 / K),
        })

    return {
        "version": 1,
        "d": int(batch.d),
        "snr_db": float(batch.snr_db),
        "sweep": sweep_records,
    }
