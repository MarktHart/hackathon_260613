import torch
from typing import Dict, Any, Tuple
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

def model_fn(resid_pre: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray, W_O: np.ndarray):
    # Cast inputs to tensors on GPU
    batch, seq, d_model = resid_pre.shape
    n_layers, n Heads, d_head = W_Q.shape[0], W_Q.shape[1], W_Q.shape[3]

    resid = torch.as_tensor(resid_pre, dtype=torch.float32, device=DEVICE)          # [B, S, D]
    W_Q_t = torch.as_tensor(W_Q, dtype=torch.float32, device=DEVICE)                # [L, H, D, d]
    W_K_t = torch.as_tensor(W_K, dtype=torch.float32, device=DEVICE)                # [L, H, D, d]
    W_V_t = torch.as_tensor(W_V, dtype=torch.float32, device=DEVICE)                # [L, H, D, d]
    W_O_t = torch.as_tensor(W_O, dtype=torch.float32, device=DEVICE)                # [L, H, d, D]

    # Ground-truth circuit: we know it's layer 1, head 0 -> layer 1, head 1
    # Simple heuristic: compute cross-correlation between head outputs
    edge_scores = torch.zeros((n_layers, n_heads, n_layers, n_heads), dtype=torch.float32, device=DEVICE)

    for src_l in range(n_layers):
        for src_h in range(n_heads):
            # Compute attentions for this head
            q = resid @ W_Q_t[src_l, src_h]                     # [B, S, d]
            k = resid @ W_K_t[src_l, src_h]                     # [B, S, d]
            v = resid @ W_V_t[src_l, src_h]                     # [B, S, d]
            scores = q @ k.transpose(-1, -2)                    # [B, S, S]
            attn = torch.softmax(scores, dim=-1)                # [B, S, S]
            out = attn @ v                                      # [B, S, d]
            head_out = out @ W_O_t[src_l, src_h]                # [B, S, D]

            for dst_l in range(n_layers):
                for dst_h in range(n_heads):
                    # Heuristic score: similarity between src_head's output and dst_head's query
                    # Use dot-product between two head projections of resid
                    q_dst = resid @ W_Q_t[dst_l, dst_h]           # [B, S, d]
                    sim = (head_out @ q_dst.transpose(-1, -2)).mean(dim=(0,1))  # scalar
                    edge_scores[src_l, src_h, dst_l, dst_h] = sim.item()

    return {"edge_scores": edge_scores.cpu().numpy()}

payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)