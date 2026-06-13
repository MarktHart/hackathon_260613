"""attention_sort / first_pass — hand-built rank-routing sorting head.

Mechanism (a minimal delta from base_model.py's single Attention block):

  1. RANK by comparison.  A value v_j's sorted rank is the number of other
     values strictly below it:  rank_j = sum_k step(v_j - v_k).  We realise
     `step` as a sharp logistic — sigmoid(tau * (v_j - v_k)) — i.e. a
     uniform-weight attention head that reads the pairwise comparison feature
     (v_j - v_k) and *counts*.  This is value-driven, not position-driven.

  2. ROUTE by rank.  Output slot i should read the i-th smallest value, i.e.
     the position whose rank == i.  We score key j for query slot i with
     logit[i, j] = -beta * (rank_j - i)^2.  Argmax over j picks the position
     whose rank is closest to i, which is exactly argsort(values)[i].

Because both steps depend only on *value comparisons* (never on absolute
position), the same fixed (tau, beta) generalise across every length — that is
the whole point of `sort_robustness`.  A position-only shortcut cannot do this.

All compute runs in torch on CUDA (hard requirement).  task.py hands NumPy in
and expects NumPy out, so we convert at the boundary only.
"""
import argparse
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; never fall back to CPU
task = load_task(__file__)


def sorting_head_logits(values: np.ndarray, tau: float = 1.0e4,
                        beta: float = 30.0) -> np.ndarray:
    """The hand-built sorting head.  NumPy[B, L] -> NumPy[B, L, L] logits.

    Row i of each [L, L] block is the attention logits from output slot i
    (which should read the i-th smallest value) over the L input positions.
    """
    v = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)   # [N, L]
    N, L = v.shape

    # --- Step 1: soft rank via pairwise comparison (a counting head) ---
    # diff[n, j, k] = v_j - v_k
    diff = v[:, :, None] - v[:, None, :]                              # [N, L, L]
    # sigmoid(0) = 0.5 on the diagonal; subtract it so a position never counts
    # itself.  Sum over k -> approximate integer rank of position j.
    soft_rank = torch.sigmoid(tau * diff).sum(dim=2) - 0.5            # [N, L]

    # --- Step 2: route output slot i to the position with rank i ---
    idx = torch.arange(L, device=DEVICE, dtype=torch.float32)        # [L]
    # logit[n, i, j] = -beta * (rank_j - i)^2
    logits = -beta * (soft_rank[:, None, :] - idx[None, :, None]) ** 2  # [N, L, L]
    return logits.detach().cpu().numpy()


def make_model_fn(tau: float = 1.0e4, beta: float = 30.0):
    def model_fn(values: np.ndarray) -> np.ndarray:
        return sorting_head_logits(values, tau=tau, beta=beta)
    return model_fn


def _accuracy_at(values: np.ndarray, tau: float, beta: float) -> float:
    """Argmax-key accuracy of the head on `values` for given temperature."""
    logits = sorting_head_logits(values, tau=tau, beta=beta)
    argmax_key = np.argmax(logits, axis=2)                # [N, L]
    target_key = np.argsort(values, axis=1)               # [N, L]
    return float(np.mean(argmax_key == target_key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=1.0e4,
                    help="comparison sharpness for the counting head")
    ap.add_argument("--beta", type=float, default=30.0,
                    help="routing sharpness (rank -> output slot)")
    args = ap.parse_args()

    run_dir = results_dir(__file__)

    # ---- Canonical benchmark ----
    payload = task.evaluate(make_model_fn(tau=args.tau, beta=args.beta))

    # ---- Artefacts for the Demo tab ----
    batch = task.generate(seed=task.EVAL_SEED)

    # (a) A sample attention heatmap per length (sequence 0).
    samples = {}
    for L in batch.lengths:
        vals = batch.sequences[L][:1]                      # [1, L]
        logits = sorting_head_logits(vals, tau=args.tau, beta=args.beta)[0]
        z = logits - logits.max(axis=1, keepdims=True)
        attn = np.exp(z)
        attn = attn / attn.sum(axis=1, keepdims=True)
        samples[str(L)] = {
            "values": vals[0].tolist(),
            "attn": attn.tolist(),
            "target_key": np.argsort(vals[0]).tolist(),
        }
    (run_dir / "samples.json").write_text(json.dumps(samples))

    # (b) Temperature sweep at the canonical length: accuracy vs tau.
    #     Shows the mechanism is a sharp-comparison limit, and degrades
    #     gracefully as the counting head goes soft.
    canon_vals = batch.sequences[task.CANONICAL_LENGTH]
    taus = [0.5, 2.0, 8.0, 32.0, 128.0, 512.0, 2.0e3, 1.0e4]
    tau_sweep = [
        {"tau": t, "accuracy": _accuracy_at(canon_vals, t, args.beta)}
        for t in taus
    ]
    (run_dir / "tau_sweep.json").write_text(json.dumps(tau_sweep))

    record_benchmark(__file__, run_dir, payload)

    print(f"[attention_sort/first_pass] run_dir={run_dir}")
    for rec in payload["sweep"]:
        print(f"  L={rec['length']:>3}  sort_acc={rec['sort_accuracy']:.3f}  "
              f"target_mass={rec['target_mass']:.3f}  "
              f"out_sorted={rec['output_sortedness']:.3f}")


if __name__ == "__main__":
    main()
