import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)

# Hand-built AND mechanism using a query and key projection.
# task.evaluate calls model_fn(q_A (d,), q_B (d,), residual (n_positions, d))
# and expects per-position logits (n_positions,).
d = 64
w_q = np.eye(d)          # Identity-like query projection that passes feature directions A, B
w_k = np.eye(d) * 0.9    # Slightly decayed key projection to encourage selective attention
scale_factor = np.sqrt(d)  # Standard attention sqrt(d_head) scaling


def model_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    Compute per-position AND logits via a bilinear conjunction of the two probe
    projections: (residual @ w_q @ q_A) * (residual @ w_k @ q_B) / sqrt(d_head).

    q_A, q_B: Shape (d,)
    residual: Shape (n_positions, d)
    Returns:  Shape (n_positions,)
    """
    q_At = torch.as_tensor(q_A, dtype=torch.float32, device=DEVICE)
    q_Bt = torch.as_tensor(q_B, dtype=torch.float32, device=DEVICE)
    rt = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    w_q_t = torch.as_tensor(w_q, dtype=torch.float32, device=DEVICE)
    w_k_t = torch.as_tensor(w_k, dtype=torch.float32, device=DEVICE)

    a = (rt @ w_q_t) @ q_At        # (n,) query-side projection onto feature A
    b = (rt @ w_k_t) @ q_Bt        # (n,) key-side projection onto feature B
    logits = (a * b) / float(scale_factor)  # AND-like conjunction, scaled
    return logits.detach().cpu().numpy()


payload = task.evaluate(model_fn)

run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
