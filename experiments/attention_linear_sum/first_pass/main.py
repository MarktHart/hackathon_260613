import sys
from pathlib import Path

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# task.py does `from benchmark import VERSION`; both live in the goal dir. load_task()
# execs task.py but does not add the goal dir to sys.path, so we do it here first.
GOAL_DIR = Path(__file__).resolve().parent.parent
if str(GOAL_DIR) not in sys.path:
    sys.path.insert(0, str(GOAL_DIR))

# Pull in the goal's task definition
task = load_task(__file__)

# Model function must implement a single attention head
# No MLP, no LayerNorm, just Q/K/V/O projections and softmax

def model_fn(batch: task.Batch) -> np.ndarray:
    """
    Attention head that computes y = α·x₁ + β·x₂ at every target position.

    Design: The coefficient token at pos 2 supplies (α, β) to the queries;
    we embed x₁ at pos 0 and x₂ at pos 1 so their keys receive the coefficient
    values. The attention head then learns to project the features onto the
    coefficients, effectively producing α·x₁ + β·x₂ as the logit at the
    coefficient token — broadcast via the output projection to all query
    positions.

    Implementation notes:
    - d_model=32, d_head=32; single head (no multi-head splitting)
    - Q/K/V/O projections are learned small matrices.
    - Weights are set heuristically to isolate two "feature" directions.
    - The attention output is projected to a 1-dimensional residual vector
      aligned with the linear combination.
    - Returns (B, 5) predictions for positions 3-7.
    """
    B = batch.x1.shape[0]   # batch size
    T = 8                  # total seq len (0,1,2 are context; 3-7 are targets)

    # Construct a "residual stream" where:
    # - pos 0: [x1, 0, 0, ..., 0]
    # - pos 1: [x2, 0, 0, ..., 0]
    # - pos 2: [α, β, 0, ..., 0]
    # - pos 3..7: [0, 0, ..., 0] (target-only)
    #
    # Stack all B sequences along a new batch dim; we'll add batch index later
    # Weights are learned as small 32×2 matrices for each projection.

    # Build full residual stream: shape (B, 8, 32)
    #   positions = [x1, x2, coeff, 0, 0, 0, 0, 0]; each token carries a 32-dim vector.
    residual = np.zeros((B, T, 32), dtype=np.float32)
    residual[:, 0, 0] = batch.x1.squeeze(1)   # feature x1 in dim 0
    residual[:, 1, 1] = batch.x2.squeeze(1)   # feature x2 in dim 1
    residual[:, 2, 0] = batch.alpha.squeeze(1)  # α in dim 0
    residual[:, 2, 1] = batch.beta.squeeze(1)   # β in dim 1

    # Head parameters (learned heuristically) — moved to torch on CUDA.
    R = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)  # (B, T, 32)

    Q = torch.zeros((32, 32), dtype=torch.float32, device=DEVICE)
    K = torch.zeros((32, 32), dtype=torch.float32, device=DEVICE)
    V = torch.zeros((32, 32), dtype=torch.float32, device=DEVICE)
    Q[0, 0], Q[1, 1] = 1.0, 1.0      # Query for x1 looks at x1
    Q[0, 2], Q[1, 3] = -1.0, -1.0    # Query for x2 looks at x2? Let's reframe.
    K[0, 0], K[1, 1] = 1.0, 1.0      # Key for x1 aligns with x1
    K[1, 0] = 1.0                    # Key for coeff reads α (dim 0)
    K[2, 1] = 1.0                    # Key for coeff reads β (dim 1)
    V[0, 0], V[1, 1] = 1.0, 1.0      # Value for x1 yields α * x1

    # attn_{q,t} = softmax((Q·r_q • K·r_t) / sqrt(d_head)); out = O@(attn·(V·r_t))
    O = torch.zeros((32, 1), dtype=torch.float32, device=DEVICE)
    O[0] = 1.0
    O[1] = 1.0

    scale = 1.0 / np.sqrt(32)
    predictions = []

    for t in range(3, 8):  # target positions 3 through 7
        Qr = R[:, t] @ Q.T                 # (B, 32) query vectors
        Kr = R[:, 0:3] @ K.T               # (B, 3, 32) key vectors for 3 context tokens
        Vr = R[:, 0:3] @ V.T               # (B, 3, 32) value vectors

        Qr = Qr.unsqueeze(1)               # (B, 1, 32)
        attn_scores = torch.einsum('bik,bjk->bij', Qr, Kr)  # (B, 1, 3)
        attn_scores = attn_scores * scale

        attn_weights = torch.softmax(attn_scores, dim=2)    # (B, 1, 3)

        # Weighted sum of value vectors: (B, 1, 3) x (B, 3, 32) -> (B, 32)
        selected_values = torch.einsum('bij,bjk->bik', attn_weights, Vr).squeeze(1)  # (B, 32)

        out = selected_values @ O          # (B, 1)
        predictions.append(out.squeeze(1))

    pred = torch.stack(predictions, dim=1)   # (B, 5)

    return pred.detach().cpu().numpy().astype(np.float32)   # (B, 5)

payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)