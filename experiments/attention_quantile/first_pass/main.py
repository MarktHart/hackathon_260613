import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)

# ---- method: dot-product attention with per-condition temperature ----
# The goal contract is model_fn(queries, keys, scale) -> (n_queries, n_keys)
# attention weights, softmax-normalised per query. We compute the standard
# scaled dot-product logits between the fixed query/key unit vectors and apply
# the per-condition temperature `scale`: a higher scale sharpens the softmax
# (heavy-tail regime), a lower scale flattens it (light-tail regime). All
# compute runs on CUDA. A numerically stable softmax keeps every weight finite.


def model_fn(queries: np.ndarray, keys: np.ndarray, scale: float) -> np.ndarray:
    """queries: (n_q, d) float32, keys: (n_k, d) float32, scale: float.
    returns: (n_q, n_k) float32 attention weights, rows sum to 1."""
    q = torch.as_tensor(queries, dtype=torch.float32, device=DEVICE)
    k = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
    logits = (q @ k.transpose(0, 1)) * float(scale)        # (n_q, n_k)
    attn = torch.softmax(logits, dim=1)                     # stable softmax
    return attn.detach().cpu().numpy().astype(np.float32)


# ---- benchmark: evaluate the method against the canonical sweep ----
run_dir = results_dir(__file__)
payload = task.evaluate(model_fn=model_fn)
record_benchmark(__file__, run_dir, payload)
