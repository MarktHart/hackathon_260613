import numpy as np
import torch

DEVICE = "cuda"

d = 64
scale = np.sqrt(d)

# Build fixed orthogonal probe directions for A and B.
def basis_vectors(d):
    # Random orthogonal matrix via QR decomposition on random Gaussian matrix.
    # Shape (d, d) orthogonal basis.
    mat = torch.randn(d, d, device=DEVICE)
    mat = mat @ torch.diag(torch.tensor([1.0, 0.98] + [-0.01] * (d-2), device=DEVICE))
    _, q = torch.linalg.qr(mat)
    # Keep the sign pattern consistent across all runs.
    signs = torch.sign(q[0, :])
    q = q * signs
    return q.T  # (d, # dims) where dims = 2 for A and B

w_q = basis_vectors(d)[:, :2]          # (d, 2) query side for A and B
w_k = basis_vectors(d)[:, :2]          # (d, 2) key side for A and B

def model_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    Compute per-position AND logits using two probe directions (A and B).

    The head uses separate query and key projections for A and B, then forms:
        score = (residual @ w_q[:, 0] * residual @ w_q[:, 1])   (AND-like product)
                / sqrt(d)
    where residual @ w_q[:, i] projects the query residual onto feature i.
    A strong score appears only when residuals contribute on both dimensions.

    Args:
        q_A:      (d,) feature vector for A (orthogonal / aligned to first probe)
        q_B:      (d,) feature vector for B (orthogonal / aligned to second probe)
        residual: (n_positions, d) query-side residual stream

    Returns:
        attn_logits: (n_positions,) unnormalised attention logits
    """
    d_t = torch.as_tensor(d, dtype=torch.float32, device=DEVICE)

    # Convert inputs to CUDA tensors.
    q_At = torch.as_tensor(q_A, dtype=torch.float32, device=DEVICE)
    q_Bt = torch.as_tensor(q_B, dtype=torch.float32, device=DEVICE)
    rt = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)

    # Project the residual stream onto the two feature directions.
    # (n_positions,) per-position score for A and for B respectively.
    proj_A = rt @ q_At          # (n_positions,) residual projected onto q_A
    proj_B = rt @ q_Bt          # (n_positions,) residual projected onto q_B

    # AND-like bilinear conjunction: high score only when both directions fire.
    logits = (proj_A * proj_B) / torch.sqrt(d_t)

    return logits.detach().cpu().numpy()

if __name__ == "__main__":
    # Load task and evaluate.
    from agentic.experiments import load_task, record_benchmark, results_dir

    task = load_task(__file__)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)