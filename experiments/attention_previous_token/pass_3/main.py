"""Third-pass attempt: hand-built synthetic previous-token head on GPU.

Builds a single attention head that attends sharply from query i to key i-1.
The goal contract is `model_fn(residual: (L, d)) -> (L, L) logits`; the
evaluator applies the causal mask and row-wise softmax itself, so the head only
needs to produce *finite* logits whose row maximum sits at the previous token.

We score each (query i, key j) pair by the dot product of residual_i against a
copy of the residual stream shifted forward by one position, so the score is
largest when j = i-1 (a previous-token pattern read from the positional signal
in the residual). No -inf entries are emitted, keeping every logit finite even
for row 0; the evaluator's causal mask handles the future. All compute on CUDA.
"""

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

SEQ_LEN = 64
D = 64
NOISE_SWEEP = [0.0, 0.25, 0.5, 1.0, 2.0]


def build_previous_token_model_fn():
    """Return a model_fn implementing a previous-token head on the GPU.

    Signature matches task.py: (L, d) residual -> (L, L) logits.
    """

    def model_fn(residual: np.ndarray) -> np.ndarray:
        resid = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
        L, d = resid.shape

        # Key for position j is the residual one position ahead (resid[j+1]);
        # then dot(resid_i, key_j) peaks at j+1 = i  ->  j = i-1.
        key = torch.zeros_like(resid)
        key[:-1] = resid[1:]

        # Temperature sharpens the previous-token peak. Finite everywhere.
        logits = (resid @ key.transpose(0, 1)) * 4.0  # (L, L)

        return logits.detach().cpu().numpy().astype(np.float32)

    return model_fn


def main() -> None:
    task = load_task(__file__)
    model_fn = build_previous_token_model_fn()

    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    sweep = payload["sweep"]
    canonical_rec = next(r for r in sweep if r["noise"] == 0.0)
    print(f"Done. Results in {run_dir}")
    print(f"Previous-token attention (canonical): {canonical_rec['prev_token_attention']:.4f}")
    print(f"Self-attention mass: {canonical_rec['self_attention']:.4f}")
    print(f"Two-back attention mass: {canonical_rec['two_back_attention']:.4f}")
    print(f"Uniform baseline: {canonical_rec['uniform_baseline']:.6f}")


if __name__ == "__main__":
    main()
