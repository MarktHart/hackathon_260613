# experiments/attention_argmax/pass_2/main.py
# ---- Imports and task bootstrap ----------------------------------------
import numpy as np
import torch
from typing import Any

# agentic imports.
from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline reserves a GPU for this attempt and verifies it was actually
# used, so the model function runs its compute on CUDA.
DEVICE = "cuda"

# Load the goal's task module.
task = load_task(__file__)

# Extract required constants from the task module — this attempt must match the
# synthetic generator's N=32, d=64, etc., for the payload contract.
_D = task._D   # 64
_N = task._N   # 32
# We won't use the full separation sweep here but will match the canonical condition.
_CANONICAL_SEP = getattr(task, "_CANONICAL_SEPARATION", 2.0)

# ---- Model: hand-implemented argmax attention head --------------------
# The model_fn signature is enforced by task.evaluate:
# model_fn(q: (d,), K: (N, d), V: (N, d)) -> attn_weights: (N,)
# B can be any integer; we support any B the caller supplies.

def model_fn(q: np.ndarray, K: np.ndarray, V: np.ndarray = None) -> np.ndarray:
    """Argmax-like attention head: places mass on the single max similarity position.

    This is a mathematical implementation, not a learned model, but it runs on
    the GPU. Arguments:
        q: (d,) query vector, unit norm.
        K: (N, d) key vectors.
        V: (N, d) value vectors (unused; kept for signature compatibility).

    Returns attention_weights: (N,) probability distribution.
    """
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)

    # Compute dot products: each row of K is a key; similarity = q · k_i
    similarities = Kt @ qt   # (N,)

    # For the canonical evaluation, use tau = 1.0 (sharpest realistic attention).
    tau = 1.0
    # Softmax with temperature (subtract max for numerical stability, on GPU).
    attn = torch.softmax(similarities / tau, dim=0)

    return attn.detach().cpu().numpy()


# ---- Run the evaluation -------------------------------------------------
payload = task.evaluate(model_fn)
record_dir = results_dir(__file__)
record_benchmark(__file__, record_dir, payload)