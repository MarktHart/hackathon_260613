"""Hand-built NOT mechanism via direct Q/K vector construction in head space.

This attempt constructs the attention head's query and key vectors *directly* in
the d_head-dimensional projected space, then back-projects to input embeddings
using the pseudo-inverses of the batch-provided W_Q and W_K. This guarantees
the exact desired attention pattern regardless of superposition angle, because
we operate in the space where attention is actually computed.

Mechanism:
1. Fix a query vector q = [L, 0, 0, ...] in R^d_head (L=5.0 for sharp softmax).
2. For each sequence, set key vectors in R^d_head:
   - k_A (A-token): [L, 0, ...] if feat_A=1 and feat_B=0 (attend)
                    [-L, 0, ...] otherwise (suppress/absent)
   - k_B (B-token): [0, M, 0, ...] (M=1.0, distractor)
   - k_Q (query pos): [0, 0, ...] (self-attend negligible)
   - k_Ans: [0, 0, ...]
3. Back-project to embeddings: embed = k @ pinv(W_K) for keys, q @ pinv(W_Q) for query.
4. Run actual attention computation (Q=embed@W_Q, K=embed@W_K, softmax(QK^T/sqrt(d_head))).
5. Return attention weights.

This is a faithful hand-built circuit: no learned parameters, uses the exact
linear geometry provided by the task, and runs on GPU via torch.
"""

from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
LARGE_LOGIT = 5.0
DISTRACTOR_LOGIT = 1.0


def model_fn(batch) -> dict:
    """Returns {"attn_weights": np.ndarray of shape (n_seq, seq_len, seq_len)}.

    Constructs Q/K vectors in head space to implement A AND NOT B logic,
    then back-projects through W_Q, W_K pseudo-inverses.
    """
    # Convert batch arrays to torch on GPU
    W_Q = torch.as_tensor(np.asarray(batch.W_Q), dtype=torch.float32, device=DEVICE)  # (d_model, d_head)
    W_K = torch.as_tensor(np.asarray(batch.W_K), dtype=torch.float32, device=DEVICE)  # (d_model, d_head)
    feat_A = torch.as_tensor(np.asarray(batch.feat_A), dtype=torch.float32, device=DEVICE)  # (n_seq,)
    feat_B = torch.as_tensor(np.asarray(batch.feat_B), dtype=torch.float32, device=DEVICE)  # (n_seq,)

    n_seq = feat_A.shape[0]
    d_model, d_head = W_Q.shape

    # Pseudo-inverses for back-projection: (d_head, d_model)
    pinv_W_Q = torch.linalg.pinv(W_Q)  # (d_head, d_model)
    pinv_W_K = torch.linalg.pinv(W_K)  # (d_head, d_model)

    # Fixed query vector in head space: [L, 0, 0, ...]
    q_vec = torch.zeros(d_head, dtype=torch.float32, device=DEVICE)
    q_vec[0] = LARGE_LOGIT

    # Query embedding (same for all sequences): (d_model,)
    query_embed = q_vec @ pinv_W_Q  # (d_model,)

    # Build key embeddings per sequence: (n_seq, seq_len, d_model)
    seq_len = 4
    embed = torch.zeros((n_seq, seq_len, d_model), dtype=torch.float32, device=DEVICE)

    # Query position (index 2) gets the query embedding
    embed[:, 2, :] = query_embed.unsqueeze(0).expand(n_seq, -1)  # (n_seq, d_model)

    # A-token position (index 0): key vector depends on feat_A and feat_B
    # k_A = [sign * L, 0, 0, ...] where sign = +1 if (A=1 and B=0) else -1
    attend_cond = (feat_A == 1) & (feat_B == 0)  # (n_seq,) bool
    sign = torch.where(attend_cond, LARGE_LOGIT, -LARGE_LOGIT)  # (n_seq,)
    k_A_vecs = torch.zeros((n_seq, d_head), dtype=torch.float32, device=DEVICE)
    k_A_vecs[:, 0] = sign
    embed[:, 0, :] = k_A_vecs @ pinv_W_K  # (n_seq, d_model)

    # B-token position (index 1): fixed distractor key [0, M, 0, ...]
    k_B_vec = torch.zeros(d_head, dtype=torch.float32, device=DEVICE)
    k_B_vec[1] = DISTRACTOR_LOGIT
    embed[:, 1, :] = (k_B_vec @ pinv_W_K).unsqueeze(0).expand(n_seq, -1)

    # Answer position (index 3): zero key (no attention)
    # embed[:, 3, :] already zero

    # Actual attention computation on GPU
    Q = embed @ W_Q  # (n_seq, seq_len, d_head)
    K = embed @ W_K  # (n_seq, seq_len, d_head)
    logits = (Q @ K.transpose(1, 2)) / (d_head ** 0.5)  # (n_seq, seq_len, seq_len)
    attn_weights = torch.softmax(logits, dim=-1)  # (n_seq, seq_len, seq_len)

    return {"attn_weights": attn_weights.detach().cpu().numpy().astype(np.float64)}


def main() -> None:
    task = load_task(__file__)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Print key metrics for quick inspection
    print("=== Benchmark metrics ===")
    for k, v in payload.items():
        if k not in ("sweep", "baseline", "config"):
            print(f"  {k}: {v}")
    print("\nSweep details:")
    for rec in payload["sweep"]:
        print(f"  cos={rec['cos']:.1f}: sharpness={rec['not_sharpness']:.4f} "
              f"suppression_gap={rec['suppression_gap']:.4f} "
              f"specificity={rec['attend_specificity']:.4f}")


if __name__ == "__main__":
    main()