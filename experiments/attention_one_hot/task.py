import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    """Container for the fixed synthetic data across all sequence lengths."""
    # For each length L, we store (keys, values, target_pos, query)
    # keys: (L, d_model), values: (L, d_model), target_pos: int, query: (d_model,)
    data_by_length: dict[int, tuple[np.ndarray, np.ndarray, int, np.ndarray]]
    temperature: float
    d_model: int
    canonical_length: int

def generate(seed: int = 0) -> Batch:
    """
    Deterministically generate the one-hot attention task.

    For each sequence length L in the sweep:
    - Pick a random target position using a per-length RNG stream.
    - Construct keys: one target key = query, others = random orthogonal vectors.
    - Construct values: one-hot at target position (standard basis vector).
    - Query is a fixed random unit vector (same across lengths).

    The seed controls all randomness. Same seed → same Batch.
    """
    rng = np.random.default_rng(seed)
    d_model = 32
    temperature = 0.1
    canonical_length = 64
    lengths = [16, 32, 64, 128, 256]

    # Fixed query vector (unit norm)
    query = rng.normal(size=d_model)
    query = query / np.linalg.norm(query)

    data_by_length = {}
    for L in lengths:
        # Derive a per-length RNG so changing the sweep doesn't shift earlier lengths
        length_rng = np.random.default_rng(seed * 1000 + L)

        target_pos = int(length_rng.integers(0, L))

        # Build keys: target key = query, others = random orthogonal to query
        keys = np.zeros((L, d_model), dtype=np.float32)
        keys[target_pos] = query.astype(np.float32)

        # Fill other positions with random vectors orthogonal to query
        for i in range(L):
            if i == target_pos:
                continue
            v = length_rng.normal(size=d_model).astype(np.float32)
            # Project out query component
            v = v - np.dot(v, query) * query
            norm = np.linalg.norm(v)
            if norm > 1e-8:
                v = v / norm
            keys[i] = v

        # Values: a distinct random unit vector per position (a fixed value
        # codebook). Because every position carries its own direction, the
        # attention output direction reveals *where* attention was placed — so
        # output_cosine is a non-degenerate end-to-end check. (One-hot values
        # would make the output direction invariant to the weight distribution,
        # collapsing output_cosine to a constant.)
        values = np.zeros((L, d_model), dtype=np.float32)
        for i in range(L):
            vv = length_rng.normal(size=d_model).astype(np.float32)
            vnorm = np.linalg.norm(vv)
            values[i] = vv / vnorm if vnorm > 1e-8 else vv

        data_by_length[L] = (keys, values, target_pos, query.astype(np.float32))

    return Batch(
        data_by_length=data_by_length,
        temperature=temperature,
        d_model=d_model,
        canonical_length=canonical_length,
    )

def evaluate(model_fn: Callable[[np.ndarray, np.ndarray, float], np.ndarray]) -> dict:
    """
    Run model_fn over all sequence lengths in the sweep.

    Args:
        model_fn: Callable with signature (query, keys, temperature) -> attn_weights.
            It returns the attention weight distribution over positions, shape
            (L,). This *is* the attempt's attention pattern — every metric below
            is derived from it, so different attempts produce different numbers.

    Returns:
        Payload dict matching the contract in README.md.
    """
    batch = generate(seed=0)
    sweep_records = []

    for L in [16, 32, 64, 128, 256]:
        keys, values, target_pos, query = batch.data_by_length[L]

        # The attempt returns its attention distribution over the L positions.
        attn_weights = np.asarray(
            model_fn(query, keys, batch.temperature), dtype=np.float64
        )
        if attn_weights.shape != (L,):
            raise ValueError(
                f"model_fn must return attention weights of shape ({L},) for "
                f"sequence length {L}, got {attn_weights.shape}"
            )

        # Metrics derived directly from the attempt's attention distribution.
        target_attention = float(attn_weights[target_pos])
        peak_attention = float(np.max(attn_weights))
        # Entropy in nats (clip to non-negative for numerical safety; a valid
        # distribution is already non-negative).
        w_nn = np.clip(attn_weights, 0.0, None)
        entropy = float(-np.sum(w_nn * np.log(w_nn + 1e-12)))

        # End-to-end output: weighted sum of the per-position value codebook.
        # Aligns with the target's value vector only when attention concentrates
        # on the target position.
        output = attn_weights @ values  # (d_model,)
        true_value = values[target_pos]
        output_norm = np.linalg.norm(output)
        true_norm = np.linalg.norm(true_value)
        if output_norm > 1e-8 and true_norm > 1e-8:
            output_cosine = float(np.dot(output, true_value) / (output_norm * true_norm))
        else:
            output_cosine = 0.0

        sweep_records.append({
            "length": L,
            "target_pos": target_pos,
            "attention_entropy": entropy,
            "peak_attention": peak_attention,
            "target_attention": target_attention,
            "output_cosine": output_cosine,
        })

    return {
        "version": 1,
        "canonical_length": batch.canonical_length,
        "temperature": batch.temperature,
        "d_model": batch.d_model,
        "sweep": sweep_records,
    }

def random_model_fn() -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    """
    Returns a dummy model_fn that emits a random (normalised) attention
    distribution of the right shape. Pure NumPy, no torch, no GPU. Used for the
    smoke test — it should score near the uniform baseline, not pass.
    """
    def _random_fn(query: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
        rng = np.random.default_rng(42)
        w = rng.random(keys.shape[0])
        return w / w.sum()
    return _random_fn