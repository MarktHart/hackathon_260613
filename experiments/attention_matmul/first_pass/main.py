import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees a visible GPU for this process.
DEVICE = "cuda"

def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Produce the ground-truth attribution matrix: true attention (softmax(QK^T/√d))

    The true attention score is `Q @ K.T / sqrt(d_head)`, softmax-normalised along
    the key dimension (j), then softmax-normalised. This is the same pathway
    that generates `true_attn` inside task.py. Returning it lets the scorer compare
    the explanation against the known computational pathway.

    Returns (batch, n_heads, seq_len, seq_len) of the true attention probabilities.
    """
    B, H, T, D = Q.shape
    assert K.shape == Q.shape, "Q and K must have matching shapes"
    assert V.shape == Q.shape, "V must have the same (batch, n_heads, T, D) shape"

    # Convert to device tensors
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    scale = 1.0 / torch.sqrt(torch.tensor(D, device=DEVICE))

    scores = Qt @ Kt.transpose(-1, -2) * scale
    attn = torch.softmax(scores, dim=-1)

    return attn.detach().cpu().numpy()

if __name__ == "__main__":
    task = load_task(__file__)
    run_dir = results_dir(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)