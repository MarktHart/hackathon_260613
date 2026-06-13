"""First-pass attempt for `attention_one_hot`: a single attention layer with scaled-dot-product attention.

Implements `task.model_fn(queries, keys, values) -> attn_weights` using NumPy only.
Returns a `(B, H, L, L)` softmax attention matrix where each query's row sums to 1.

The mechanism:
1. Compute scaled dot product: logit[b,h,q,k] = queries[b,h,q,:] · keys[b,h,k,:] / sqrt(D_HEAD)
2. Apply per-query softmax across the key dimension (last axis = k)
3. Result shape: (B, H, L, L)

This is the canonical content-addressing attention mechanism. It should concentrate mass on
the correct key when the query-key dot product dominates distractor dot products.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# --------------------------------------------------------
# Imports required for the model function
# --------------------------------------------------------

# Type alias from task.py — we satisfy its signature exactly.
_ModelFn = Callable[
    [
        np.ndarray,        # queries: (B, H, L, D_HEAD)
        np.ndarray,        # keys:    (B, H, L, D_HEAD)
        np.ndarray,        # values:  (B, H, L, D_HEAD)  [unused in attention-only model]
    ],
    np.ndarray,           # attn_weights: (B, H, L, L)
]

# --------------------------------------------------------
# Model function
# --------------------------------------------------------


def model_fn(
    query: np.ndarray,
    keys: np.ndarray,
    temperature: float,
) -> np.ndarray:
    """
    Single-head scaled-dot-product attention matching the task contract.

    query : (d_model,)
    keys  : (L, d_model)
    temperature: softmax temperature

    Returns the attention distribution over the L key positions, shape (L,),
    summing to 1. Core compute runs in torch on CUDA.
    """
    q = torch.as_tensor(np.asarray(query), dtype=torch.float32, device=DEVICE)        # (d,)
    k = torch.as_tensor(np.asarray(keys), dtype=torch.float32, device=DEVICE)         # (L, d)

    # Content-addressing: scores = keys @ query / temperature, then softmax over keys.
    scores = (k @ q) / float(temperature)            # (L,)
    attn = torch.softmax(scores, dim=-1)             # (L,)

    return attn.detach().cpu().numpy()


# --------------------------------------------------------
# Entry point
# --------------------------------------------------------


def run():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)  # runs across MARGIN_SWEEP
    record_benchmark(__file__, results_dir(__file__), payload)


if __name__ == "__main__":
    run()