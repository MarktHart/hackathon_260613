import numpy as np
import torch
from typing import Optional
from agentic.experiments import load_task, record_benchmark, results_dir

# Pipeline guarantees GPU visible
DEVICE = "cuda"

# Load the synthetic generator and evaluator
task = load_task(__file__)


def model_fn(q: np.ndarray, k: np.ndarray, v: np.ndarray,
             logit_scale: float, causal_mask: Optional[np.ndarray]) -> dict:
    """
    Compute exact softmax attention on GPU and return saturation metrics.
    
    Args:
        q: query vectors, shape (batch, seq_len, d_head)
        k: key vectors, shape (batch, seq_len, d_head)
        v: value vectors, shape (batch, seq_len, d_head)
        logit_scale: scalar multiplier for logits before softmax
        causal_mask: optional boolean mask (seq_len, seq_len)
    
    Returns:
        dict with:
            - 'attn_weights': (batch, seq_len, seq_len)
            - 'attn_entropy': (batch, seq_len)
            - 'saturation_score': float (higher = more saturated)
    """
    batch, seq_len, d_head = q.shape
    
    # Move to GPU
    q_t = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    k_t = torch.as_tensor(k, dtype=torch.float32, device=DEVICE)
    
    if causal_mask is not None:
        causal_mask_t = torch.as_tensor(causal_mask, dtype=torch.bool, device=DEVICE)
    else:
        causal_mask_t = None
    
    # Compute logits: (batch, seq_len, seq_len)
    logits = torch.einsum('bqd,bkd->bqk', q_t, k_t) * logit_scale
    
    # Apply causal mask if provided
    if causal_mask_t is not None:
        logits = torch.where(causal_mask_t, logits, torch.full_like(logits, -1e9))
    
    # Softmax
    attn_weights = torch.softmax(logits, dim=-1)
    
    # Per-query entropy
    eps = 1e-12
    attn_entropy = -torch.sum(attn_weights * torch.log(attn_weights + eps), dim=-1)
    
    # Saturation score: mean max attention weight (higher = more saturated)
    # Also works well: 1 - entropy / log(seq_len) 
    max_attn_weight = attn_weights.max(dim=-1).values.mean().item()
    
    # Alternative: normalized saturation (0 = uniform, 1 = one-hot)
    # max_entropy = np.log(seq_len)
    # saturation_score = 1.0 - (attn_entropy.mean().item() / max_entropy)
    
    # Use max_attn_weight as saturation_score - it's monotonic with saturation
    saturation_score = max_attn_weight
    
    return {
        'attn_weights': attn_weights.detach().cpu().numpy().astype(np.float32),
        'attn_entropy': attn_entropy.detach().cpu().numpy().astype(np.float32),
        'saturation_score': float(saturation_score),
    }


def main():
    # Run the task with our model function
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)


if __name__ == "__main__":
    main()