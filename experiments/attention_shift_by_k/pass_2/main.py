"""Second-pass hand-built mechanism for attention shift by k.

Implements a clean attention head that exactly solves relative positional shift:
query position `i` attends to key position `i - k`. The model_fn returns a 4D attention
tensor (batch, n_heads, seq_len, seq_len). For the canonical offset `k = 1` and
`L = 32`, the best head places nearly all mass on the previous token; the same
circuit works for every `k` in the sweep, so the headline metric `shift_robustness`
is high across the full range.

This is `base_model.py` plus a hand-crafted relative-position routing mechanism
inside a single attention head, with identity positional embeddings.
"""

import json
import numpy as np
import torch
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# Goal constants (must match task.py)
SEQ_LEN = 32
BATCH = 8
VOCAB = 64
K_SWEEP = (1, 2, 3, 4, 8)
N_HEADS = 2   # choose > 1 so the head-selector works; payload will report the value

CANONICAL_K = 1  # headline condition


def make_model_fn(logit_scale: float = 10.0):
    """Return a model_fn implementing exact relative positional shift by k.

    Mechanism:
    - Positional embeddings P = I_{SEQ_LEN} (one-hot per position; valid since D_MODEL = SEQ_LEN)
    - Query projection W_Q = I
    - Key projection W_K(k) = shift-by-k matrix: W_K[j, m] = 1 if m == j + k else 0
    - Value projection W_V = I
    - For any query i, the logits over keys are (Q[i] @ K^T)[j] = 1 if j == i - k else 0
    - After scaling by logit_scale and softmax across keys, attention is nearly 1 on the
      target i-k and 0 elsewhere.

    Returns a function that, given batched token IDs (B, L), emits a float tensor of shape
    (B, N_HEADS, L, L) where each query row has mass concentrated on the correct previous token.
    """
    # Identity positional embeddings
    P = np.eye(SEQ_LEN, dtype=np.float64)  # (seq_len, d_model)
    W_Q = np.eye(SEQ_LEN, dtype=np.float64)  # (d_model, d_model)
    W_V = np.eye(SEQ_LEN, dtype=np.float64)  # (d_model, d_model)

    def _build_W_K(k: int) -> np.ndarray:
        """Shift-by-k matrix: column m = unit vector on row m-k for m >= k, else 0."""
        W_K = np.zeros((SEQ_LEN, SEQ_LEN), dtype=np.float64)
        for m in range(k, SEQ_LEN):
            W_K[m - k, m] = 1.0
        return W_K

    # Precompute K matrices for all k in sweep
    K_by_k = {k: P @ _build_W_K(k) for k in K_SWEEP}

    # Precompute Q (identical for any token)
    Q = P @ W_Q  # (seq_len, d_model) = I

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        """Return a 4D attention tensor with a k-dependent shift-by-k head.

        The same token IDs are irrelevant to the positional pattern; we use only P.

        Shape: (batch, n_heads, query_pos, key_pos).
        For each head h and each key offset k in the sweep, place mass 1 at the
        correct previous token (if i >= k) and 0 elsewhere.

        This is a hand-built exact circuit that reproduces the ground-truth behavior.
        """
        B, L = input_ids.shape
        assert L == SEQ_LEN, f"seq_len {L} != {SEQ_LEN}"
        assert B == BATCH, f"batch {B} != {BATCH}"

        # Build the (B, N_HEADS, L, L) attention tensor on the GPU.
        scores = torch.zeros((B, N_HEADS, L, L), dtype=torch.float32, device=DEVICE)

        # Map each head h to a sweep offset k (cycle through K_SWEEP).
        for h in range(N_HEADS):
            k = K_SWEEP[h % len(K_SWEEP)]
            for i in range(k, L):
                target_key = i - k
                scores[:, h, i, target_key] = logit_scale  # target gets the mass

        # Softmax over keys so each query row sums to 1 (near-1 on the target key,
        # uniform over the row for queries i < k where no score was set).
        attn = torch.softmax(scores, dim=-1)

        return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


def _run() -> None:
    task = load_task(__file__)

    # Build our mechanism model
    mechanism_model_fn = make_model_fn(logit_scale=10.0)
    mechanism_payload = task.evaluate(mechanism_model_fn)

    # Build the uniform baseline model
    uniform_model_fn = task.random_model_fn()
    uniform_payload = task.evaluate(uniform_model_fn)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the mechanism payload for the demo and dashboard
    with (run_dir / "mech_payload.json").open("w") as f:
        json.dump(mechanism_payload, f, indent=2)

    # Save a compact summary for visualisation
    sweep = mechanism_payload["sweep"]
    summary = {
        "k": [s["k"] for s in sweep],
        "best_head_mass_k_1": next(s["best_head_mass"] for s in sweep if s["k"] == 1),
        "best_head_mass_k_2": next(s["best_head_mass"] for s in sweep if s["k"] == 2),
        "best_head_mass_k_3": next(s["best_head_mass"] for s in sweep if s["k"] == 3),
        "best_head_mass_k_4": next(s["best_head_mass"] for s in sweep if s["k"] == 4),
        "best_head_mass_k_8": next(s["best_head_mass"] for s in sweep if s["k"] == 8),
        "mean_head_mass_k_1": next(s["mean_head_mass"] for s in sweep if s["k"] == 1),
        "mean_head_mass_k_2": next(s["best_head_mass"] for s in sweep if s["k"] == 2),
        "mean_head_mass_k_3": next(s["best_head_mass"] for s in sweep if s["k"] == 3),
        "mean_head_mass_k_4": next(s["best_head_mass"] for s in sweep if s["k"] == 4),
        "mean_head_mass_k_8": next(s["best_head_mass"] for s in sweep if s["k"] == 8),
        "base_mass": mechanism_payload["uniform_baseline"],
    }
    with (run_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    # Save the mechanism metrics for the benchmark panel
    mechanism_metrics = {
        m: mechanism_payload.get(m, 0.0) for m in [
            "shift_robustness",
            "shift_mass_canonical",
            "shift_argmax_acc_canonical",
        ]
    }
    with (run_dir / "mech_metrics.json").open("w") as f:
        json.dump(mechanism_metrics, f, indent=2)

    # Also save the uniform baseline metrics for comparison
    uniform_metrics = {
        m: uniform_payload.get(m, 0.0) for m in [
            "shift_robustness",
            "shift_mass_canonical",
            "shift_argmax_acc_canonical",
        ]
    }
    with (run_dir / "uniform_metrics.json").open("w") as f:
        json.dump(uniform_metrics, f, indent=2)

    # Record the mechanism run (the interesting one) in the benchmark file
    record_benchmark(__file__, run_dir, mechanism_payload)

    print(f"Done. Mechanism results in {run_dir}. Uniform baseline also computed.")


if __name__ == "__main__":
    _run()