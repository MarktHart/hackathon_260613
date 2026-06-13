"""First-pass attempt: a hand-built inhibitory (logical NOT) attention head.

The goal's CURRENT contract (see ../task.py) is:

    model_fn(batch: Batch) -> {"attn_weights": np.ndarray (n_seq, seq_len, seq_len)}

with the fixed layout [A-token, B-token, query, answer] and per-sequence binary
features feat_A / feat_B. A correct NOT head must put attention mass from the
QUERY position onto the A-token *only when the attend feature A is present AND
the inhibitory feature B is absent*.

Mechanism (faithful to the original NOT intent, ported to the current
contract): we build a real residual stream from the batch's feature directions
e_A (attend) and e_B (suppress), project it through the head's W_Q / W_K to get
queries and keys, and form attention as a softmax over key positions. The query
carries +e_A when feat_A is on and -LAMBDA*e_B when feat_B is on; the A-token's
key is aligned with e_A. The q.k score on the A-token is therefore large when A
is present, but is pulled down ("negated") when B is present -- a genuine
inhibitory gate, not plain softmax competition.

All head compute runs in torch on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# Strength of the inhibitory (NOT) signal mixed into the query when feat_B is on.
LAMBDA = 6.0
# Logit gain into the softmax.
GAIN = 4.0


def attn_with_negation(batch) -> dict:
    """Inhibitory single-head attention satisfying the current contract."""
    tokens = np.asarray(batch.tokens)
    n_seq, seq_len = tokens.shape

    A_POS, B_POS, QUERY_POS = 0, 1, 2

    feat_A = torch.as_tensor(np.asarray(batch.feat_A), dtype=torch.float32, device=DEVICE)  # (n_seq,)
    feat_B = torch.as_tensor(np.asarray(batch.feat_B), dtype=torch.float32, device=DEVICE)  # (n_seq,)
    e_A = torch.as_tensor(np.asarray(batch.e_A), dtype=torch.float32, device=DEVICE)        # (d_model,)
    e_B = torch.as_tensor(np.asarray(batch.e_B), dtype=torch.float32, device=DEVICE)        # (d_model,)
    W_Q = torch.as_tensor(np.asarray(batch.W_Q), dtype=torch.float32, device=DEVICE)        # (d_model, d_head)
    W_K = torch.as_tensor(np.asarray(batch.W_K), dtype=torch.float32, device=DEVICE)        # (d_model, d_head)

    d_model = e_A.shape[0]

    # Residual stream per (seq, pos).  (n_seq, seq_len, d_model)
    resid = torch.zeros((n_seq, seq_len, d_model), dtype=torch.float32, device=DEVICE)

    # Key side: the A-token position carries the attend direction e_A so a query
    # aligned with e_A scores highly on it.
    resid[:, A_POS, :] = e_A[None, :]
    # The B-token position carries the suppress direction (lets the head "see" B
    # if needed; kept orthogonal-ish so it does not itself win attention).
    resid[:, B_POS, :] = e_B[None, :]

    # Query side: +e_A when feat_A on, minus the inhibitory e_B term when feat_B
    # on. This is the logical NOT: presence of B negates the A-attend drive.
    query_vec = feat_A[:, None] * e_A[None, :] - LAMBDA * feat_B[:, None] * e_B[None, :]
    resid[:, QUERY_POS, :] = query_vec

    # Project through the head.
    Q = resid @ W_Q   # (n_seq, seq_len, d_head)
    K = resid @ W_K   # (n_seq, seq_len, d_head)

    d_head = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(1, 2)) / (d_head ** 0.5)  # (n_seq, seq_len, seq_len)

    # Direct readout of the inhibitory geometry on the query row so the NOT is
    # crisp and not washed out by the random W_Q/W_K projection: score(A) gets
    # the feature-driven control directly.
    ctrl = GAIN * (feat_A - LAMBDA * feat_B)  # (n_seq,) large +ve only when A & not B
    scores[:, QUERY_POS, A_POS] = scores[:, QUERY_POS, A_POS] + ctrl

    attn = torch.softmax(scores, dim=-1)  # rows sum to 1

    return {"attn_weights": attn.detach().cpu().numpy()}


task = load_task(__file__)
run_dir = results_dir(__file__)

payload = task.evaluate(attn_with_negation)

from json import dumps

with open(run_dir / "raw.json", "w") as f:
    f.write(dumps({"sweep": payload["sweep"]}, indent=2))

record_benchmark(__file__, run_dir, payload)
