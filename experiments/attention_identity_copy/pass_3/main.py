from agentic.experiments import load_task, record_benchmark, results_dir
from dataclasses import dataclass
import numpy as np
import torch

DEVICE = "cuda"

# ----------------------------------------------------------
# Load the goal's definitions
# ----------------------------------------------------------
task = load_task(__file__)

@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray  # shape (B, L), int32

@dataclass(frozen=True)
class ModelOutput:
    attn_weights: np.ndarray  # shape (B, H, L, L), float32
    values: np.ndarray        # shape (B, H, L, D), float32

# ----------------------------------------------------------
# identity_copy_head(batch: Batch) -> ModelOutput
# ----------------------------------------------------------
def identity_copy_head(batch: Batch) -> ModelOutput:
    B, L = batch.tokens.shape
    H, D = task.H, task.D

    tokens = torch.as_tensor(batch.tokens, dtype=torch.long, device=DEVICE)  # (B, L)

    # Simulate token embeddings: one-hot per token (256-dim), then project to D per head.
    embeds = torch.eye(256, dtype=torch.float32, device=DEVICE)[tokens]  # (B, L, 256)

    # Deterministic per-head value projector (256 -> D). Same projector reused for
    # Q/K/V; only V is used downstream as `values`.
    g = torch.Generator(device=DEVICE).manual_seed(0)
    w_v = torch.randn(H, 256, D, generator=g, device=DEVICE, dtype=torch.float32) / 256.0

    proj_v = torch.einsum("bld,hdk->bhlk", embeds, w_v)  # (B, H, L, D)
    v = proj_v.reshape(B, H, L, D)                        # (B, H, L, D)

    # Identity head: uniform diagonal attention (all positions equally likely).
    attn_weights = torch.eye(L, dtype=torch.float32, device=DEVICE) / L
    attn_weights = attn_weights[None, None, :, :].expand(B, H, L, L).contiguous()

    # Final output (computed for the mechanism; not part of ModelOutput).
    _ = torch.einsum("bhij,bhjk->bhik", attn_weights, v)  # (B, H, L, D)

    return ModelOutput(
        attn_weights=attn_weights.detach().cpu().numpy(),
        values=v.detach().cpu().numpy(),
    )

# ----------------------------------------------------------
# Run, record, and persist
# ----------------------------------------------------------
payload = task.evaluate(identity_copy_head)

run_dir = results_dir(__file__)
run_dir.mkdir(parents=True, exist_ok=True)
with (run_dir / "payload.json").open("w") as f:
    import json
    json.dump(payload, f, indent=2)

record_benchmark(__file__, run_dir, payload)
print(f"benchmark recorded to {run_dir / 'benchmark.json'}")