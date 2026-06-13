import numpy as np
import torch

DEVICE = "cuda"

def distance_head(seq_len):
    # Implement a head that focuses on nearby tokens.
    # The strength of attention falls off with distance:
    # exp(-distance/lambda) for distance > 0, and 1 for distance == 0.
    lambda_param = 4.0  # Controls how fast attention falls off

    # --- GPU compute (torch on CUDA) ---
    i = torch.arange(seq_len, device=DEVICE).view(-1, 1).to(torch.float32)
    j = torch.arange(seq_len, device=DEVICE).view(1, -1).to(torch.float32)
    dist = i - j
    attn = torch.exp(-dist / lambda_param)  # dist==0 -> exp(0)==1.0

    causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=DEVICE))
    attn = attn * causal_mask
    row_sums = attn.sum(dim=1, keepdim=True)
    attn = attn / (row_sums + (row_sums == 0).to(attn.dtype))

    return attn[None, None, :, :]

def model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    """Return attention weights that depend on positional distance.

    The task expects a dict with key "attention" of shape (n_layers, n_heads,
    S, S) (broadcast over batch) using the canonical 4 layers x 8 heads.
    """
    seq_len = input_ids.shape[1]

    # Canonical architecture for this goal.
    n_layers, n_heads = 4, 8

    attn = distance_head(seq_len)  # (1, 1, S, S) torch tensor on CUDA

    # Duplicate the same attention pattern across all layers and heads.
    attn_out = attn.expand(n_layers, n_heads, seq_len, seq_len).contiguous()

    return {"attention": attn_out.detach().cpu().numpy()}

# Run the task evaluation
if __name__ == "__main__":
    print("Running task evaluation...")
    from agentic.experiments import load_task, record_benchmark, results_dir

    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)
    print("Task evaluation completed.")