"""Pass 2: explicit context-dependent suppression NOT head, on the current contract.

The goal's CURRENT contract (../task.py) is:

    model_fn(batch: Batch) -> {"attn_weights": np.ndarray (n_seq, seq_len, seq_len)}

layout [A-token, B-token, query, answer], with per-sequence binary features
feat_A (attend) and feat_B (suppress). The original pass-2 idea -- actively
*lower* the target score when the inhibitory marker is present, rather than
relying on softmax competition -- is preserved: we subtract a scaled inhibitory
control from the query->A-token score whenever feat_B is on.

All compute runs in torch on CUDA.
"""

import numpy as np
import torch
from dataclasses import dataclass

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)


@dataclass
class Config:
    attend_gain: float = 4.0   # drive toward A when feat_A present
    suppress: float = 8.0      # strength of the active NOT suppression on feat_B


def model_fn(batch) -> dict:
    """Single inhibitory attention head with explicit target suppression."""
    cfg = Config()
    tokens = np.asarray(batch.tokens)
    n_seq, seq_len = tokens.shape

    A_POS, B_POS, QUERY_POS = 0, 1, 2

    feat_A = torch.as_tensor(np.asarray(batch.feat_A), dtype=torch.float32, device=DEVICE)
    feat_B = torch.as_tensor(np.asarray(batch.feat_B), dtype=torch.float32, device=DEVICE)
    e_A = torch.as_tensor(np.asarray(batch.e_A), dtype=torch.float32, device=DEVICE)
    e_B = torch.as_tensor(np.asarray(batch.e_B), dtype=torch.float32, device=DEVICE)
    W_Q = torch.as_tensor(np.asarray(batch.W_Q), dtype=torch.float32, device=DEVICE)
    W_K = torch.as_tensor(np.asarray(batch.W_K), dtype=torch.float32, device=DEVICE)

    d_model = e_A.shape[0]

    resid = torch.zeros((n_seq, seq_len, d_model), dtype=torch.float32, device=DEVICE)
    resid[:, A_POS, :] = e_A[None, :]
    resid[:, B_POS, :] = e_B[None, :]
    resid[:, QUERY_POS, :] = feat_A[:, None] * e_A[None, :]

    Q = resid @ W_Q
    K = resid @ W_K
    d_head = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(1, 2)) / (d_head ** 0.5)

    # Standard attend drive toward the A-token, then ACTIVE suppression: lower
    # the query->A score by suppress * feat_B. This is the context-dependent NOT
    # -- the A-token score itself drops when the inhibitory feature is present.
    base = cfg.attend_gain * feat_A
    suppression = cfg.suppress * feat_B
    scores[:, QUERY_POS, A_POS] = scores[:, QUERY_POS, A_POS] + base - suppression

    attn = torch.softmax(scores, dim=-1)
    return {"attn_weights": attn.detach().cpu().numpy()}


payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
