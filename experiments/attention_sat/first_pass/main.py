import numpy as np
import torch
from dataclasses import dataclass

# Pipeline guarantees GPU visible
DEVICE = "cuda"

# Task imports are from the generated task module
from agentic.experiments import load_task, record_benchmark, results_dir

# Load the synthetic generator and evaluator
task = load_task(__file__)


def model_fn(batch: dataclass) -> Dict[str, Any]:
    """Compute exact attention statistics on GPU.
    
    Args:
        batch: dataclass with tokens, W_Q, W_K, W_V, config
    
    Returns:
        dict with attn_entropy, attn_max, attn_top1_frac, attn_topk_frac, head_labels
    """
    # 1. Move data to GPU
    d_model = batch.tokens.shape[1]  # 64
    n_heads = batch.W_Q.shape[0]     # 4
    d_head = batch.W_Q.shape[1]      # 16

    # PyTorch requires contiguous dims for einsum with batch dims
    # We flatten to (n_heads, L, d_head) then later sum over L
    tokens_pt = torch.as_tensor(batch.tokens, dtype=torch.float32, device=DEVICE)
    W_Q_pt = torch.as_tensor(batch.W_Q, dtype=torch.float32, device=DEVICE)
    W_K_pt = torch.as_tensor(batch.W_K, dtype=torch.float32, device=DEVICE)
    W_V_pt = torch.as_tensor(batch.W_V, dtype=torch.float32, device=DEVICE)

    L = batch.tokens.shape[0]

    # Project queries and keys (n_heads, L, d_head)
    Q = torch.einsum('hdi,li->hli', W_Q_pt, tokens_pt)
    K = torch.einsum('hdi,li->hli', W_K_pt, tokens_pt)

    # Compute attention scores: (h, L, L)
    scale = torch.as_tensor(1.0 / np.sqrt(d_head), device=DEVICE)
    scores = torch.einsum('hli,hji->hlj', Q, K) * scale

    # Softmax across keys dimension
    attn = torch.softmax(scores, dim=-1)   # (n_heads, L, L)

    # Now compute per-head statistics averaged over query positions (L dim)
    entropy = -torch.mean(attn * torch.log(attn + 1e-12), dim=-1)   # (n_heads, L)
    attn_max = torch.max(attn, dim=-1).values  # (n_heads, L)

    # Top-1 fraction is just attn_max
    top1_frac = torch.mean(attn_max, dim=-1)  # (n_heads,)

    # Top-k fraction (k=4)
    k = min(4, L)
    # For top-k we sort descending and sum first k
    attn_sorted, _ = torch.sort(attn, dim=-1, descending=True)
    topk_sum = torch.sum(attn_sorted[:, :, :k], dim=-1)  # (n_heads, L)
    topk_frac = torch.mean(topk_sum, dim=-1)  # (n_heads,)

    # Return NumPy arrays — task.evaluate expects NumPy
    head_labels = [f"head_{i}" for i in range(n_heads)]

    return {
        "attn_entropy": entropy.mean(dim=-1).cpu().numpy(),   # mean over L
        "attn_max": attn_max.mean(dim=-1).cpu().numpy(),
        "attn_top1_frac": top1_frac.cpu().numpy(),
        "attn_topk_frac": topk_frac.cpu().numpy(),
        "head_labels": head_labels,
    }


def main():
    # Run the task with our model function
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)


if __name__ == "__main__":
    main()