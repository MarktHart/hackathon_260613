import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline reserves one CUDA device; use it.
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Hand-built attention mechanism for proper graph coloring.
#
# Inputs: `adj` (symmetric, zero-diagonal, float32) and `feats` (n x k+1).
# Output: (n, n) attention matrix, rows non-negative (isolated nodes are zeros).
#
# Strategy:
# - The first k columns of `feats` are one-hot colour indicators.
# - We construct a fixed projector matrix P of shape (k, k) where:
#     P[i, j] = 1.0 if i != j (different colours), else 0.0 (same colour).
#   This means color vectors c_i and c_j yield c_i @ P @ c_j = 1.0 iff colours differ.
# - Query: Q = feats[:, :k] @ P  (n, k)
# - Key:   K = feats[:, :k]      (n, k)  — plain one-hot
# - Raw scores S = Q @ K.T  (n, n) gives 1.0 for different-colour pairs, 0 otherwise.
# - Mask with adjacency so mass concentrates on edges (which are always cross-colour
#   in a proper coloring). Multiply elementwise: S * adj.
# - Add a small constant to avoid all-zero rows, then row-normalise so non-isolated
#   rows sum to 1. Isolated-node rows stay zero.
# - All tensor work happens on `cuda` to satisfy the GPU guard; final result is
#   returned as NumPy for task.evaluate's signature.
# ---------------------------------------------------------------------------

def build_projector(k: int) -> torch.Tensor:
    """Return (k, k) matrix with 1.0 off-diagonal, 0.0 on diagonal."""
    P = torch.ones((k, k), dtype=torch.float32, device=DEVICE)
    P.fill_diagonal_(0.0)
    return P


def model_fn(adj: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """
    Hand-coded attention that respects proper graph coloring structure.
    Runs on CUDA; returns NumPy (n, n) as required by task.evaluate.
    """
    n = adj.shape[0]
    k = feats.shape[1] - 1  # last column is normalised degree

    # Move to CUDA
    adj_t = torch.as_tensor(adj, dtype=torch.float32, device=DEVICE)
    color_feats = torch.as_tensor(feats[:, :k], dtype=torch.float32, device=DEVICE)

    # Fixed projector: 1 for different colours, 0 for same
    P = build_projector(k)

    # Q = color_feats @ P  -> each row is sum of other colours' one-hots
    # K = color_feats       -> one-hot rows
    Q = color_feats @ P                  # (n, k)
    K = color_feats                      # (n, k)

    # S_ij = 1.0 if i and j have different colours, else 0.0
    S = Q @ K.T                          # (n, n)

    # Mask with adjacency: only edges get attention mass.
    # In a proper coloring, all edges are cross-color, so S_ij=1 on edges.
    S = S * adj_t

    # Row-normalise. Add epsilon so isolated nodes (degree 0) don't divide by zero;
    # we'll zero their rows afterwards anyway.
    row_sums = S.sum(dim=1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    attn = S / row_sums

    # Zero out isolated-node rows explicitly (degree == 0)
    degrees = adj_t.sum(dim=1)
    attn = attn * (degrees > 0).float().unsqueeze(1)

    return attn.detach().cpu().numpy().astype(np.float32)


def main():
    """
    Entry point. Loads the task (data and evaluator), hands the hand-built model
    function to it, gets a payload, and records it in the results directory.
    """
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)


if __name__ == "__main__":
    main()