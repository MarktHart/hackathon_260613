from dataclasses import dataclass
import numpy as np
from typing import Callable, Any

@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray      # int32[B, L]
    targets: np.ndarray     # int32[B]

# Canonical model architecture constants
N_LAYERS = 2
N_HEADS = 4
N_HEADS_TOTAL = N_LAYERS * N_HEADS  # 8
VOCAB_SIZE = 128
SEQ_LEN = 64
BATCH_SIZE = 256
GROUND_TRUTH_INDUCTION_HEADS = 2  # one per layer

# Deterministic synthetic data (same every run)
def _make_canonical_batch(seed: int = 0) -> Batch:
    rng = np.random.default_rng(seed)
    # Generate random sequences
    tokens = rng.integers(0, VOCAB_SIZE, size=(BATCH_SIZE, SEQ_LEN), dtype=np.int32)
    # Targets: token at position (delay) before the last position
    # Use a fixed delay of 5 for canonical condition
    delay = 5
    target_pos = SEQ_LEN - 1
    source_pos = target_pos - delay
    targets = tokens[:, source_pos].copy()
    return Batch(tokens=tokens, targets=targets)

# Load the canonical checkpoint weights (embedded as numpy arrays for portability)
# In reality this would load a .pt file; here we embed the exact weights the
# canonical model was trained to. The random_model_fn ignores these.
def _load_canonical_weights() -> dict:
    # Placeholder: in the real scaffold this loads from a shipped .npy/.pt file.
    # For the smoke test we don't need real weights because random_model_fn
    # returns zeros. The evaluator only checks shapes.
    return {}

def generate(seed: int = 0) -> Batch:
    """Deterministic canonical batch. Seed is accepted but ignored (fully fixed)."""
    return _make_canonical_batch(0)

def random_model_fn() -> Callable[[Batch], dict[str, np.ndarray]]:
    """Returns a model_fn that outputs zero attention weights of correct shape."""
    def _fn(batch: Batch) -> dict[str, np.ndarray]:
        B, L = batch.tokens.shape
        # Shape: [B, n_layers, n_heads, L, L]
        attn = np.zeros((B, N_LAYERS, N_HEADS, L, L), dtype=np.float32)
        return {"attn_weights": attn}
    return _fn

def evaluate(model_fn: Callable[[Batch], dict[str, np.ndarray]]) -> dict:
    """
    Run model_fn on the canonical batch, compute per-head induction scores,
    and return the payload dict expected by benchmark.score.
    """
    batch = generate()
    out = model_fn(batch)
    
    # Validate output
    attn = out["attn_weights"]
    assert attn.shape == (BATCH_SIZE, N_LAYERS, N_HEADS, SEQ_LEN, SEQ_LEN), \
        f"attn_weights shape {attn.shape} != expected"
    assert attn.dtype == np.float32
    
    # Compute per-head induction score:
    # For each head, average attention weight from target_pos to source_pos
    # across the batch. This is a proxy for "does this head attend to the copy source?".
    delay = 5
    target_pos = SEQ_LEN - 1
    source_pos = target_pos - delay
    
    # attn[b, layer, head, target_pos, source_pos]
    scores = attn[:, :, :, target_pos, source_pos].mean(axis=0)  # [n_layers, n_heads]
    # Normalize to [0, 1] per head by dividing by max possible (1.0)
    # Already in [0,1] since attention weights are softmax outputs.
    per_head_scores = scores.flatten().tolist()  # layer-major order, length 8
    
    # Threshold sweep
    thresholds = [round(t * 0.05, 2) for t in range(21)]  # 0.00, 0.05, ..., 1.00
    threshold_sweep = []
    for thr in thresholds:
        pred_count = sum(1 for s in per_head_scores if s >= thr)
        threshold_sweep.append({"threshold": thr, "predicted_count": pred_count})
    
    payload = {
        "version": 1,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "ground_truth_induction_heads": GROUND_TRUTH_INDUCTION_HEADS,
        "per_head_scores": per_head_scores,
        "threshold_sweep": threshold_sweep,
    }
    return payload