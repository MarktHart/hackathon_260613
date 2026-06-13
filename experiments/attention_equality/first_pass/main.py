import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

V = task.V  # vocabulary size


def model_fn(batch) -> np.ndarray:
    """Equality-lookup attention head.

    The original attempt targeted an obsolete permutation-equivariance contract
    (`model_fn(tokens) -> dict`). The CURRENT contract (task.evaluate /
    docstring) is:
        model_fn(batch: Batch) -> np.ndarray of shape (B, L, L), row-stochastic
        over the causally-allowed keys (mass 0 on disallowed positions).

    We implement a faithful equality head on the GPU: each query attends to
    earlier key positions holding the *same* token. We realise this with a real
    one-hot token embedding (vocab x vocab identity) so the QK score
        score[q, k] = <onehot(tok_q), onehot(tok_k)>
    is 1 iff the tokens are equal, then add a large temperature and softmax over
    the causal mask. This routes nearly all of each query's mass onto its single
    matching key, beating the uniform baseline.
    """
    tokens = np.asarray(batch.tokens)
    B, L = tokens.shape

    tok_t = torch.as_tensor(tokens, dtype=torch.int64, device=DEVICE).clamp_(0, V - 1)
    mask = torch.as_tensor(np.asarray(batch.mask), dtype=torch.bool, device=DEVICE)  # (B, L, L)

    # One-hot embedding (identity) -> equality QK score is a real GPU matmul.
    emb = torch.eye(V, dtype=torch.float32, device=DEVICE)            # (V, V)
    x = emb[tok_t]                                                    # (B, L, V)

    temperature = 30.0
    scores = torch.einsum("bqv,bkv->bqk", x, x) * temperature        # (B, L, L), =temp on matches

    neg_inf = torch.tensor(-1e9, device=DEVICE, dtype=torch.float32)
    scores = torch.where(mask, scores, neg_inf)

    attn = torch.softmax(scores, dim=-1)                             # (B, L, L)
    # Zero out disallowed keys explicitly so masked entries are exactly 0.
    attn = attn * mask.to(attn.dtype)

    attn_np = attn.detach().cpu().numpy().astype(np.float64)
    return attn_np


payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print("Done. canonical match_mass:", payload["canonical"]["match_mass"],
      "uniform_baseline:", payload["canonical"]["uniform_baseline"])
