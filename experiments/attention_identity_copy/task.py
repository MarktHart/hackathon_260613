import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray          # shape (B, L), int32

@dataclass(frozen=True)
class ModelOutput:
    attn_weights: np.ndarray    # shape (B, H, L, L), float32
    values: np.ndarray          # shape (B, H, L, D), float32

ModelFn = Callable[[Batch], ModelOutput]

# Canonical configuration
B = 32
L = 16
H = 8
D = 64
SWEEP_TOKENS = [0, 64, 128, 192, 255]
CANONICAL_TOKEN = 128

def generate(seed: int = 0) -> Batch:
    """Deterministic batch of random tokens. Seed is used but sweep overrides tokens."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, 256, size=(B, L), dtype=np.int32)
    return Batch(tokens=tokens)

def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn on sweep batches, compute copy fidelity and diagonal attention mass."""
    sweep_records = []
    for token in SWEEP_TOKENS:
        # Build batch where every position is the sweep token
        tokens = np.full((B, L), token, dtype=np.int32)
        batch = Batch(tokens=tokens)
        out = model_fn(batch)

        # Validate shapes
        if out.attn_weights.shape != (B, H, L, L):
            raise ValueError(f"attn_weights shape {out.attn_weights.shape} != ({B},{H},{L},{L})")
        if out.values.shape != (B, H, L, D):
            raise ValueError(f"values shape {out.values.shape} != ({B},{H},{L},{D})")

        # Attention output = attn_weights @ values  (sum over source positions)
        # shape: (B, H, L, D)
        attn_out = np.einsum('bhij,bhjd->bhid', out.attn_weights, out.values)

        # For identity copy, we want output at position i to match value at position i
        # Cosine similarity per (batch, head, position)
        # Normalize along D dimension
        eps = 1e-8
        attn_out_norm = attn_out / (np.linalg.norm(attn_out, axis=-1, keepdims=True) + eps)
        values_norm = out.values / (np.linalg.norm(out.values, axis=-1, keepdims=True) + eps)
        cos_sim = np.sum(attn_out_norm * values_norm, axis=-1)  # (B, H, L)

        # Mean over batch and positions → per-head fidelity
        fidelity_per_head = cos_sim.mean(axis=(0, 2))  # (H,)
        best_head = int(np.argmax(fidelity_per_head))
        best_fidelity = float(fidelity_per_head[best_head])

        # Diagonal attention mass for best head
        diag_attn = out.attn_weights[:, best_head, np.arange(L), np.arange(L)]  # (B, L)
        diag_mass = float(diag_attn.mean())

        sweep_records.append({
            "token": int(token),
            "copy_fidelity": best_fidelity,
            "diag_attn_mass": diag_mass,
            "best_head": best_head,
        })

    return {
        "version": 1,
        "sweep": sweep_records,
        "canonical_token": CANONICAL_TOKEN,
        "config": {
            "B": B, "L": L, "H": H, "D": D,
            "sweep_tokens": SWEEP_TOKENS,
        },
    }

def random_model_fn() -> ModelFn:
    """Return a model_fn that produces random valid outputs (for smoke test)."""
    def _fn(batch: Batch) -> ModelOutput:
        B_, L_ = batch.tokens.shape
        # Random attention weights: uniform over source positions (rows sum to 1)
        attn = np.ones((B_, H, L_, L_), dtype=np.float32) / L_
        # Random values
        vals = np.random.default_rng(42).normal(0, 1, size=(B_, H, L_, D)).astype(np.float32)
        return ModelOutput(attn_weights=attn, values=vals)
    return _fn