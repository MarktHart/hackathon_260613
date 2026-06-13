import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# ----- types -------------------------------------------------------------

@dataclass(frozen=True)
class Batch:
    """
    Pre-generated query/key pairs for the cosine sweep.
    queries: (n_bins * pairs_per_bin, d_model)
    keys:    (n_bins * pairs_per_bin, d_model)
    cosines: (n_bins * pairs_per_bin,)  -- cosine label for each pair
    bin_starts: (n_bins + 1,) -- row indices where each bin starts in the flat arrays
    """
    queries: np.ndarray
    keys: np.ndarray
    cosines: np.ndarray
    bin_starts: np.ndarray


# ----- deterministic generator -------------------------------------------

def generate(seed: int = 0) -> Batch:
    """
    Generate the canonical sweep data. Deterministic for a given seed.
    Returns a Batch with 21 bins × 100 pairs = 2100 pairs in d=64.
    """
    rng = np.random.default_rng(seed)

    d_model = 64
    pairs_per_bin = 100
    cos_sweep = np.linspace(-1.0, 1.0, 21, dtype=np.float32)  # 21 values
    n_bins = len(cos_sweep)
    n_total = n_bins * pairs_per_bin

    # Pre-allocate
    queries = np.empty((n_total, d_model), dtype=np.float32)
    keys = np.empty((n_total, d_model), dtype=np.float32)
    cosines = np.empty(n_total, dtype=np.float32)
    bin_starts = np.empty(n_bins + 1, dtype=np.int32)

    row = 0
    for i, cos in enumerate(cos_sweep):
        bin_starts[i] = row
        # Generate random orthogonal basis for this bin
        # We want pairs with exactly this cosine similarity.
        # Method: pick random unit q, then construct k = cos*q + sin*ortho
        for j in range(pairs_per_bin):
            q = rng.normal(size=d_model).astype(np.float32)
            q /= np.linalg.norm(q) + 1e-8
            # Random unit vector orthogonal to q
            ortho = rng.normal(size=d_model).astype(np.float32)
            ortho -= np.dot(ortho, q) * q
            ortho /= np.linalg.norm(ortho) + 1e-8
            k = cos * q + np.sqrt(max(0.0, 1.0 - cos*cos)) * ortho
            # Renormalise for numerical safety
            k /= np.linalg.norm(k) + 1e-8
            queries[row] = q
            keys[row] = k
            cosines[row] = cos
            row += 1
    bin_starts[n_bins] = row

    return Batch(
        queries=queries,
        keys=keys,
        cosines=cosines,
        bin_starts=bin_starts,
    )


# ----- model function type -----------------------------------------------

ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""
model_fn(queries, keys) -> scores
  queries: (n_pairs, d_model) float32
  keys:    (n_pairs, d_model) float32
  returns: (n_pairs,) float32  -- logits or normalised weights
"""


# ----- evaluator ---------------------------------------------------------

def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """
    Run model_fn over the canonical batch, apply per-bin softmax if needed,
    aggregate mean/std per cosine bin, return payload dict.
    """
    batch = generate(seed=42)  # canonical seed, ignores caller's seed
    n_bins = len(batch.bin_starts) - 1
    pairs_per_bin = batch.bin_starts[1] - batch.bin_starts[0]  # 100

    # Get raw scores from model
    scores = model_fn(batch.queries, batch.keys)  # (n_total,)
    if scores.shape != (n_bins * pairs_per_bin,):
        raise ValueError(f"model_fn returned shape {scores.shape}, expected {(n_bins * pairs_per_bin,)}")

    # Map each pair's raw score to an attention weight in [0, 1].
    #
    # IMPORTANT: we do NOT softmax within a bin. Every pair in a bin shares
    # the same cosine, so a per-bin softmax would force the bin mean to
    # exactly 1/pairs_per_bin regardless of the model, erasing all signal
    # (the very thing the sweep is supposed to measure). Instead we apply a
    # per-pair sigmoid so the bin mean is free to vary with cosine and the
    # logistic-fit sharpness/threshold metrics become meaningful.
    #
    # Dual mode: if the model already returns weights in [0, 1] (e.g. a head
    # that emits probabilities), use them as-is; otherwise treat the scores
    # as logits and squash with a sigmoid. Note: an all-zeros output (the
    # provided random_model_fn) lands in [0, 1] and is read as zero-weight,
    # giving a flat, no-mechanism sweep — that is the intended smoke-test
    # signal, not an error.
    if np.all(scores >= -1e-6) and np.all(scores <= 1.0 + 1e-6):
        attention = np.clip(scores, 0.0, 1.0).astype(np.float32)
    else:
        attention = (1.0 / (1.0 + np.exp(-np.clip(scores, -30.0, 30.0)))).astype(np.float32)

    mean_attentions = np.empty(n_bins, dtype=np.float32)
    std_attentions = np.empty(n_bins, dtype=np.float32)
    cos_values = np.empty(n_bins, dtype=np.float32)

    for i in range(n_bins):
        start = batch.bin_starts[i]
        end = batch.bin_starts[i + 1]
        bin_attn = attention[start:end]  # (pairs_per_bin,)
        cos_val = batch.cosines[start]

        mean_attentions[i] = float(np.mean(bin_attn))
        std_attentions[i] = float(np.std(bin_attn, ddof=1))
        cos_values[i] = float(cos_val)

    # Build sweep records
    sweep = [
        {
            "cosine": float(cos_values[i]),
            "mean_attention": float(mean_attentions[i]),
            "std_attention": float(std_attentions[i]),
        }
        for i in range(n_bins)
    ]

    payload = {
        "version": 1,
        "config": {
            "d_model": 64,
            "pairs_per_bin": 100,
            "cosine_sweep": [float(c) for c in cos_values],
            "seed": 42,
            "normalisation": "sigmoid_per_pair",
        },
        "sweep": sweep,
        "model_info": {
            "name": "unknown",
            "type": "unknown",
            "notes": "",
        },
    }
    return payload


# ----- random model for smoke test ---------------------------------------

def random_model_fn() -> ModelFn:
    """
    Returns a ModelFn that outputs all zeros (a no-mechanism reference).
    Under the per-pair normalisation this yields a flat sweep, so the
    sharpness metric is ~0 and the attempt is correctly flagged as having
    no sign-detection mechanism. Use it only as a pipeline smoke test.
    Pure NumPy, no torch, no GPU.
    """
    def _fn(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        n_pairs = queries.shape[0]
        return np.zeros(n_pairs, dtype=np.float32)
    return _fn