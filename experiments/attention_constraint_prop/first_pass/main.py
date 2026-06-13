import sys, random, time
import torch
import numpy as np
from collections.abc import Iterator

from agentic.experiments import load_task, record_benchmark, results_dir

# ---- model: small 1-layer self-att block on GPU ------------------------------

Device = "cuda"
assert torch.cuda.is_available(), "gpuGuard should expose a GPU for this slot"

def make_net(inp_shape: tuple[int, int]) -> torch.nn.Module:
    """Return a tiny single-layer self-attention block on Device."""
    B, S = inp_shape
    C = 256               # embedding dimension
    H = 8                 # number of heads
    net = torch.nn.ModuleDict({
        "emb": torch.nn.Embedding(VOCAB_SIZE, C),
        "qkv": torch.nn.Linear(C, 3 * C // H, bias=False),   # 3 * C per head
        "atten": torch.nn.MultiheadAttention(C, H, batch_first=True, dropout=0.0),
        "out": torch.nn.Linear(C, C, bias=False),
    }).to(Device)
    return net


def model_fn(input_ids: np.ndarray) -> np.ndarray:
    """Compute self-attention weights for the given batch."""
    B, S = input_ids.shape
    net = make_net((B, S))

    # token embeddings on the GPU
    ids = torch.as_tensor(input_ids, dtype=torch.int64, device=Device)   # [B, S]
    emb = net["emb"](ids)                                               # [B, S, C]

    # query, key, value projections; each head gets C/H dims
    qkv = net["qkv"](emb)                                                # [B, S, 3 * H, C//H]
    qkv = qkv.view(B, S, 3, H, C // H).permute(2, 0, 1, 3, 4)          # [3, B, S, H, C//H]
    q, k, v = qkv.unbind(0)                                            # each [B, S, H, C//H]

    # scaled dot product attention (softmax over S)
    attn_weights = torch.einsum('bshd,bseh->bhsd', q, k)                 # [B, H, S, S]
    attn_weights = attn_weights / torch.sqrt(torch.tensor(C // H, device=Device))
    attn_weights = torch.softmax(attn_weights, dim=-1)                     # [B, H, S, S]

    # expand dims to match model_fn output shape [B, L, H, S, S]
    # L = 1 because we have one attention layer; can be changed if more layers added
    attn_weights = attn_weights.unsqueeze(1)                           # [B, 1, H, S, S]

    # move back to CPU and cast to float32 numpy array
    return attn_weights.cpu().numpy().astype(np.float32)


# ---- run ---------------------------------------------------------------------

task = load_task(__file__)
print("Computing attention weights...")   # ~300ms on a single small GPU
run_dir = results_dir(__file__)

# model_fn must do real compute on CUDA; a silent CPU fallback would fail the guard
Payload = task.evaluate(model_fn)          # single batch, canonical config
record_benchmark(__file__, run_dir, Payload)
print(f"payload: {Payload}")
print(f"Wrote benchmark results to: {run_dir}/")