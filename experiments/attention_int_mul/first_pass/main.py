"""First-pass hand-built circuit for integer multiplication routing.

This attempt decodes the operands from their fixed embeddings by nearest-neighbour
lookup in the known INT_EMBED table, computes the product, and scores each
candidate by its cosine similarity to the product's embedding. All heavy
arithmetic runs on the GPU via torch; the model_fn signature matches task.py.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

# ---------------------------------------------------------------------------
# Device: the pipeline guarantees a visible CUDA device. No CPU fallback.
# ---------------------------------------------------------------------------
DEVICE = "cuda"

# ---------------------------------------------------------------------------
# Load the task to access the fixed embedding table and evaluator.
# ---------------------------------------------------------------------------
task = load_task(__file__)
INT_EMBED = task.INT_EMBED                      # (V, d) numpy, unit vectors
V, D = INT_EMBED.shape

# Pre-compute the embedding table on the GPU for fast nearest-neighbour search.
INT_EMBED_T = torch.as_tensor(INT_EMBED, dtype=torch.float32, device=DEVICE)  # (V, d)


def decode_nearest(vec: torch.Tensor) -> int:
    """Return the integer whose fixed embedding is closest (cosine) to `vec`.

    `vec` is a 1-D torch tensor on DEVICE, shape (d,). INT_EMBED_T is (V, d).
    Since both are unit-norm, argmax of dot product = nearest neighbour.
    """
    # (V,) dot products
    sims = INT_EMBED_T @ vec
    return int(torch.argmax(sims).item())


def model_fn(a_vec: np.ndarray, b_vec: np.ndarray, key_vecs: np.ndarray) -> np.ndarray:
    """Hand-built multiplication-routing head running on the GPU.

    Steps (all in torch on CUDA):
      1. Decode operands a, b from their embeddings via nearest-neighbour.
      2. Compute product p = a * b.
      3. Fetch the product embedding phi(p).
      4. Score each candidate key by cosine similarity to phi(p).
      5. Return unnormalised logits (n_positions,) as NumPy for the evaluator.
    """
    # Move inputs to GPU tensors
    a_t = torch.as_tensor(a_vec, dtype=torch.float32, device=DEVICE)          # (d,)
    b_t = torch.as_tensor(b_vec, dtype=torch.float32, device=DEVICE)          # (d,)
    keys_t = torch.as_tensor(key_vecs, dtype=torch.float32, device=DEVICE)    # (n_pos, d)

    # 1–2. Decode operands and compute product
    a_int = decode_nearest(a_t)
    b_int = decode_nearest(b_t)
    p = a_int * b_int

    # 3. Product embedding
    p_emb = INT_EMBED_T[p]                                                    # (d,)

    # 4. Cosine similarity scores (keys are already unit-norm)
    logits_t = keys_t @ p_emb                                                 # (n_pos,)

    # 5. Back to NumPy for task.evaluate
    return logits_t.detach().cpu().numpy()


if __name__ == "__main__":
    # Evaluate and record benchmark
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Done. Results in {run_dir}")
    print(f"Mean routing accuracy: {payload['sweep'][2]['routing_accuracy']:.3f} at K=8")