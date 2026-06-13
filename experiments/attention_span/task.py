from dataclasses import dataclass
import numpy as np


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal integral of y over x. NumPy 2.0 removed np.trapz, so we
    implement it directly to stay version-independent. Returns 0.0 for fewer
    than two points."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size < 2:
        return 0.0
    dx = np.diff(x)
    return float(np.sum(dx * (y[1:] + y[:-1]) / 2.0))


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray          # (batch, seq_len), int32
    target_positions: np.ndarray   # (batch,), int32
    query_positions: np.ndarray    # (batch,), int32
    distances: np.ndarray          # (batch,), int32


def generate(seed: int = 0) -> Batch:
    """Generate needle-in-haystack sequences with targets at canonical distances.

    Deterministic for a given seed. Canonical seed is 0.
    Returns 900 examples (100 per distance) of length 512.
    """
    rng = np.random.default_rng(seed)
    batch_size = 900
    seq_len = 512
    canonical_distances = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256], dtype=np.int32)
    samples_per_distance = 100

    input_ids = rng.integers(1, 1000, size=(batch_size, seq_len), dtype=np.int32)
    target_positions = np.zeros(batch_size, dtype=np.int32)
    query_positions = np.zeros(batch_size, dtype=np.int32)
    distance_labels = np.zeros(batch_size, dtype=np.int32)

    idx = 0
    for d in canonical_distances:
        for _ in range(samples_per_distance):
            q = 0
            t = int(d)
            input_ids[idx, t] = 9999   # needle token
            input_ids[idx, q] = 8888   # query token
            target_positions[idx] = t
            query_positions[idx] = q
            distance_labels[idx] = d
            idx += 1

    return Batch(
        input_ids=input_ids,
        target_positions=target_positions,
        query_positions=query_positions,
        distances=distance_labels
    )


def evaluate(model_fn) -> dict:
    """Run model_fn on the canonical batch, compute attention on target per distance.

    Args:
        model_fn: Callable with signature (input_ids: np.ndarray) -> np.ndarray.
                  input_ids shape (batch, seq_len). Returns attention weights
                  shape (batch, num_heads, seq_len, seq_len) or (batch, seq_len, seq_len).

    Returns:
        Payload dict matching the contract in README.md.
    """
    batch = generate(seed=0)
    attn = model_fn(batch.input_ids)

    # Normalise to (batch, num_heads, seq_len, seq_len)
    if attn.ndim == 3:
        attn = attn[:, None, :, :]
    elif attn.ndim != 4:
        raise ValueError(f"model_fn must return array with 3 or 4 dimensions, got {attn.ndim}")

    batch_size, num_heads, seq_len, _ = attn.shape
    if batch_size != batch.input_ids.shape[0]:
        raise ValueError(f"Batch size mismatch: model returned {batch_size}, expected {batch.input_ids.shape[0]}")
    if seq_len != batch.input_ids.shape[1]:
        raise ValueError(f"Sequence length mismatch: model returned {seq_len}, expected {batch.input_ids.shape[1]}")

    # Average attention over heads, then extract query->target for each example
    attn_mean_heads = attn.mean(axis=1)  # (batch, seq_len, seq_len)
    attn_to_target = np.zeros(batch_size, dtype=np.float32)
    for i in range(batch_size):
        q = batch.query_positions[i]
        t = batch.target_positions[i]
        attn_to_target[i] = attn_mean_heads[i, q, t]

    # Aggregate by distance
    canonical_distances = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    sweep = []
    for d in canonical_distances:
        mask = batch.distances == d
        if mask.any():
            mean_attn = float(attn_to_target[mask].mean())
            std_attn = float(attn_to_target[mask].std())
            sweep.append({
                "distance": int(d),
                "mean_attention_on_target": mean_attn,
                "std_attention_on_target": std_attn,
                "n_samples": int(mask.sum())
            })

    # Headline AUC: trapezoidal rule in log2 space, normalised to [0, 1]
    distances_arr = np.array([s["distance"] for s in sweep], dtype=np.float64)
    attns_arr = np.array([s["mean_attention_on_target"] for s in sweep], dtype=np.float64)
    log_distances = np.log2(distances_arr)
    denom = _trapz(np.ones_like(attns_arr), log_distances)
    auc = _trapz(attns_arr, log_distances) / denom if denom > 0 else 0.0

    return {
        "version": 1,
        "canonical_seq_len": 512,
        "canonical_distances": canonical_distances,
        "samples_per_distance": 100,
        "sweep": sweep,
        "attention_span_auc": float(auc)
    }


def random_model_fn():
    """Return a random model_fn for smoke testing.

    The returned callable has signature (input_ids: np.ndarray) -> np.ndarray
    and returns uniform attention weights of shape (batch, 12, seq_len, seq_len).
    """
    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        batch_size, seq_len = input_ids.shape
        num_heads = 12
        # Uniform attention: 1/seq_len everywhere
        return np.full((batch_size, num_heads, seq_len, seq_len),
                       1.0 / seq_len, dtype=np.float32)
    return model_fn