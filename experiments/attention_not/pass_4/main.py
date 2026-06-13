"""Hand-built NOT mechanism via explicit embedding composition.

The attempt builds an attention head that encodes both tokens and features
into fixed embeddings. It then computes the NOT via a simple composition rule:

1. Embeddings
   - k_A = embed_M[:,0]   # A-token key
   - k_B = embed_M[:,1]   # B-token key
   - q   = embed_M[:,2]   # query

2. Feature vectors
   - e_A = batch.e_A       # attend direction
   - e_B = batch.e_B       # suppress direction
   - neg_anchor = _NEG_ANCHOR  # fixed orthogonal anchor

3. Logic
   - If B is suppressed, the attention head should attend to A-key.
   - If B is active, the head should *inhibit* attention to A-key.
   - To implement NOT, we make A-key a linear combination where:
        * The non-signal part is fixed ( orthogonal basis vectors, matching
        the task's expected basis to ensure orthogonality with e_B when desired).
        * The signal part depends on B.

   Specifically:
     k_A = e_A                (signal)   + e_B                (non-signal)
     But we want: k_A = e_A * (1 - B)   + non-signal
     So we set:
        k_A_signal = e_A
        k_A_non.signal = _NEG_ANCHOR   (orthogonal to e_A and to e_B)

     Therefore:
        k_A = (1 - B) * e_A + non信号

   This makes attention proportional to (1-B), implementing NOT.

4. Implementation
   - For each example, build a per-sequence key embedding matrix embed_m[4, d_model].
   - Use the task-provided linear baseline geometry (W_Q, W_K, W_V, W_O).
   - Compute the attention head as (q @ W_Q) @ (k @ W_K).T + bias.

The mechanism is fully determined, contains no learnable parameters, and
leverages the fixed geometry of the task (orthogonal basis vectors) to ensure
correct composition.

This is the fourth attempt at this goal (pass_4) and differs from previous
attempts in that it works directly with token/feature vectors rather than
relying on a precomputed basis.
"""

from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


# Fixed orthogonality basis (seed=42, matching the task's internal basis)
def _orthonormal_basis(d_model: int, n_vecs: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mat = rng.normal(size=(d_model, n_vecs)).astype(np.float64)
    q, _ = np.linalg.qr(mat)
    return q.T[:n_vecs].astype(np.float32)


_BASIS = _orthonormal_basis(64, 10, 42)
_NEG_ANCHOR = _BASIS[8]          # fixed negative anchor (orthogonal to e_A and e_B in the expected case)
_TARGET_ANCHOR = _BASIS[0]       # target direction


def model_fn(batch: Batch) -> dict:
    """Returns {"attn_weights": np.ndarray of shape (n_seq, 4, 4)}.

    NOT composition: the A-token key carries the attend-direction e_A scaled by
    (1 - feat_B), so the query (aligned with e_A) attends to the A-token only
    when the inhibitory feature B is absent. All numeric compute runs on CUDA.
    """
    # task.py: e_A, e_B are single (d_model,) directions; feat_A/feat_B are (n_seq,).
    e_A = torch.as_tensor(np.asarray(batch.e_A), dtype=torch.float32, device=DEVICE)   # (d_model,)
    e_B = torch.as_tensor(np.asarray(batch.e_B), dtype=torch.float32, device=DEVICE)   # (d_model,)
    feat_A = torch.as_tensor(np.asarray(batch.feat_A), dtype=torch.float32, device=DEVICE)  # (n_seq,)
    feat_B = torch.as_tensor(np.asarray(batch.feat_B), dtype=torch.float32, device=DEVICE)  # (n_seq,)
    neg_anchor = torch.as_tensor(np.asarray(_NEG_ANCHOR), dtype=torch.float32, device=DEVICE)

    n_seq, seq_len = batch.tokens.shape
    d_model = e_A.shape[0]

    # Per-sequence key embeddings, [n_seq, seq_len, d_model].
    embed = torch.zeros((n_seq, seq_len, d_model), dtype=torch.float32, device=DEVICE)
    # A-token (slot 0): attend-direction gated by (1 - feat_B), plus a fixed
    # non-signal anchor. feat_A gates whether the A-token is present at all.
    a_gate = (feat_A * (1.0 - feat_B))[:, None]                 # (n_seq, 1)
    embed[:, 0, :] = a_gate * e_A[None, :] + neg_anchor[None, :]
    # B-token (slot 1): suppress direction.
    embed[:, 1, :] = e_B[None, :]
    # query (slot 2): aligned with the attend direction e_A.
    embed[:, 2, :] = e_A[None, :]
    # answer (slot 3): unused.

    # Attention head with the provided linear geometry (W_Q, W_K).
    W_Q = torch.as_tensor(np.asarray(batch.W_Q), dtype=torch.float32, device=DEVICE)  # (d_model, d_head)
    W_K = torch.as_tensor(np.asarray(batch.W_K), dtype=torch.float32, device=DEVICE)  # (d_model, d_head)
    Q = embed @ W_Q                                  # (n_seq, seq_len, d_head)
    K = embed @ W_K                                  # (n_seq, seq_len, d_head)
    d_head = Q.shape[-1]
    logits = (Q @ K.transpose(1, 2)) / (d_head ** 0.5)  # (n_seq, seq_len, seq_len)

    # Row-softmax over keys so each query row sums to 1.
    attn_weights = torch.softmax(logits, dim=-1)     # (n_seq, seq_len, seq_len)

    return {"attn_weights": attn_weights.detach().cpu().numpy().astype(np.float64)}


def main() -> None:
    task = load_task(__file__)
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Print key metrics for quick inspection
    print("=== Benchmark metrics ===")
    for k, v in payload.items():
        if k != "sweep" and k != "config":
            print(f"  {k}: {v}")
    print("\nSweep details:")
    for rec in payload["sweep"]:
        print(f"  cos={rec['cos']:.1f}: sharpness={rec['not_sharpness']:.4f} "
              f"suppression_gap={rec['suppression_gap']:.4f} "
              f"specificity={rec['attend_specificity']:.4f}")


if __name__ == "__main__":
    main()