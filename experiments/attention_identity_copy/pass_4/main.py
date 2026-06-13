from agentic.experiments import load_task, record_benchmark, results_dir
from dataclasses import dataclass
import json
import numpy as np
import torch

DEVICE = "cuda"  # guaranteed by pipeline; no CPU fallback

# -----------------------------------------------------------------
# Load the task and its constants directly from task.py
# -----------------------------------------------------------------
task = load_task(__file__)

# Canonical dims (must match task.py)
B = 32
L = 16
H = 8
D = 64


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray  # shape (B, L), int32


@dataclass(frozen=True)
class ModelOutput:
    attn_weights: np.ndarray  # shape (B, H, L, L), float32
    values: np.ndarray        # shape (B, H, L, D), float32


# -----------------------------------------------------------------
# identity_copy_head(batch: Batch) -> ModelOutput
# -----------------------------------------------------------------
# Approach: a hand-built identity-copy circuit. Each head attends purely
# to the diagonal (position i -> i), so the attention output at position i
# equals the value vector at position i exactly. We give each position a
# distinct, deterministic value vector so cosine fidelity is meaningful
# (copying succeeds => fidelity ~ 1.0).
#
# The previous version crossed numpy and torch dtypes (np.eye(...,
# dtype=torch.float32, device=...)), which numpy cannot interpret. Here all
# torch ops get torch dtypes and the only numpy op (the final conversion)
# stays in numpy land.
def identity_copy_head(batch: Batch) -> ModelOutput:
    B_, L_ = batch.tokens.shape
    assert B_ == B and L_ == L, \
        f"Batch dimensions must be ({B}, {L}), got ({B_}, {L_})"

    # token ids -> int64 for safe indexing on GPU
    tokens = torch.as_tensor(batch.tokens, dtype=torch.int64, device=DEVICE)  # (B, L)

    # Diagonal attention: each query i attends only to key i.
    # eye is a torch tensor with a torch dtype (not numpy).
    eye = torch.eye(L_, dtype=torch.float32, device=DEVICE)               # (L, L)
    attn_weights = eye.view(1, 1, L_, L_).expand(B_, H, L_, L_).contiguous()  # (B, H, L, L)

    # Deterministic per-position value vectors, distinct across positions so
    # that copying position i -> i yields high cosine fidelity. We make them
    # depend on the token id and the position so all sweep tokens behave well.
    g = torch.Generator(device=DEVICE)
    g.manual_seed(0)
    # Per-head, per-token-id embedding table: (H, vocab=256, D)
    value_table = torch.randn(H, 256, D, generator=g, device=DEVICE, dtype=torch.float32)
    # Add a strong per-position component so even when every token in a batch is
    # identical, position i has a unique value (diagonal copy stays informative).
    pos_table = torch.randn(H, L_, D, generator=g, device=DEVICE, dtype=torch.float32)

    # Gather token-dependent component: (H, B, L, D)
    # tokens: (B, L) -> index into value_table[h]
    tok_vals = value_table[:, tokens, :]            # (H, B, L, D)
    tok_vals = tok_vals.permute(1, 0, 2, 3)         # (B, H, L, D)
    pos_vals = pos_table.view(1, H, L_, D)          # (1, H, L, D)
    values = tok_vals + pos_vals                     # (B, H, L, D)

    return ModelOutput(
        attn_weights=attn_weights.detach().cpu().numpy().astype(np.float32),
        values=values.detach().cpu().numpy().astype(np.float32),
    )


# -----------------------------------------------------------------
# Run, record, and persist
# -----------------------------------------------------------------
def main():
    payload = task.evaluate(identity_copy_head)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "payload.json").open("w") as f:
        json.dump(payload, f, indent=2)

    record_benchmark(__file__, run_dir, payload)
    print(f"benchmark recorded to {run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
