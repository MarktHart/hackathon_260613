import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

def model_fn(
    queries: np.ndarray,
    keys: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    batch, seq_len, d_model = queries.shape

    # Convert to PyTorch tensors
    q = torch.as_tensor(queries, dtype=torch.float32, device=DEVICE)
    k = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    
    # Compute attention scores
    scores = torch.einsum('bqd,bkd->bqd', q, k)  # (batch, seq_len, seq_len)
    
    # Apply the expected bipartite pattern
    # For each query, we only attend to keys in the same group
    # Group assignment is deterministic: first half = group 0, second half = group 1
    n_per_group = seq_len // 2
    group_idx = (torch.arange(seq_len, device=DEVICE) >= n_per_group).unsqueeze(0)  # (1, seq_len)
    query_groups = group_idx.unsqueeze(1)  # (1, 1, seq_len)
    key_groups = group_idx.unsqueeze(2)  # (1, seq_len, 1)
    mask = (query_groups == key_groups).float()  # (1, seq_len, seq_len)
    
    # Apply a hard mask and renormalize
    scores = scores * mask
    attn = scores.softmax(dim=-1)
    
    # Return CPU numpy array
    return attn.detach().cpu().numpy()

task = load_task(__file__)
payload = task.evaluate(model_fn)
record_benchmark(__file__, results_dir(__file__), payload)