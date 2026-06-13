import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# ------------------------------------------------------------------
# Goal: reconstruct composed attention matrix A_chain = A2 @ A1
#   from row-stochastic patterns A1, A2, and evaluate how robustly
#   the mechanism works as attention rows become more peaked (higher
#   performance when the single-head shortcut breaks down)
# ------------------------------------------------------------------
# First-pass attempt: a simple PyTorch function that directly computes
# the matrix product of the provided head matrices *on the GPU*.
# This tests if the exact linear algebra operation is sufficient
# to pass the composition-robustness metric.
# ------------------------------------------------------------------
# The task.py contract:
#   model_fn(A1: np.ndarray, A2: np.ndarray) -> np.ndarray
#   each (num_heads, seq_len, seq_len)
# ------------------------------------------------------------------
# GPU requirement: the pipeline guarantees CUDA_DEVICE_VISIBLE.
# We must use torch tensors on cuda for the real compute, then
# return the NumPy version. Fallback to CPU would be rejected.
# ------------------------------------------------------------------

DEVICE = "cuda"

def model_fn(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
    """Directly compute the composed attention matrix on the GPU."""
    qt = torch.as_tensor(A1, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(A2, dtype=torch.float32, device=DEVICE)
    # Matrix product per head (broadcasts across batch dimension)
    A_chain = kt @ qt  # (num_heads, seq, seq) on GPU
    return A_chain.detach().cpu().numpy()


if __name__ == "__main__":
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)