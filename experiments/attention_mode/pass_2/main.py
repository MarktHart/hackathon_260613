"""Hand-built minimum-KL template classifier for attention modes (pass_2).

Preserves the README intent: a no-learning classifier that pre-computes an ideal
template per mode and, for each head's attention matrix, picks the mode whose
template is closest in KL divergence, softmaxing the scores into a probability
over modes.

The original attempt targeted an out-of-date contract and crashed before it ran:
  - it read `task.modes` (no such attribute) and asserted it was a `list`;
    the goal exposes `task.MODES`, a *tuple* of five canonical modes
    (positional, uniform, diagonal, induction, previous_token) — note there is
    no "sink" mode and no per-offset induction variants.
  - it implemented `model_fn` for a single `(L,)` row returning a dict, but the
    current contract is `(n_heads, L, L) -> (n_heads, N_MODES)` ndarray.

This rewrite matches the current contract exactly. All arithmetic runs on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)
MODES = list(task.MODES)
N_MODES = len(MODES)
L = task.CANONICAL_L


# ---- Clean reference templates per mode (the patterns task.generate uses) ----
def _template(mode: str, L: int) -> np.ndarray:
    pat = np.zeros((L, L), dtype=np.float32)
    if mode == "positional":
        pat[:, 0] = 1.0
    elif mode == "uniform":
        pat[:] = 1.0 / L
    elif mode == "diagonal":
        pat = np.eye(L, dtype=np.float32)
    elif mode == "induction":
        for i in range(L):
            pat[i, (i + 1) % L] = 1.0
    elif mode == "previous_token":
        for i in range(L):
            pat[i, (i - 1) % L] = 1.0
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return pat


_clean = np.stack([_template(m, L) for m in MODES], axis=0)        # (M, L, L)
_clean = _clean / _clean.sum(axis=2, keepdims=True)
TEMPLATES = torch.as_tensor(_clean, dtype=torch.float32, device=DEVICE)


def model_fn(attention_matrices: np.ndarray) -> np.ndarray:
    """(n_heads, L, L) -> (n_heads, N_MODES) probability rows, on GPU."""
    pat = torch.as_tensor(attention_matrices, dtype=torch.float32, device=DEVICE)  # (H,L,L)
    eps = 1e-7

    p = pat.unsqueeze(1)                # (H, 1, L, L)
    q = TEMPLATES.unsqueeze(0)         # (1, M, L, L)
    # Row-wise KL(p || q), averaged over query rows -> (H, M).
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=3).mean(dim=2)

    # Minimum KL = best match; softmax(-KL) turns it into a distribution.
    probs = torch.softmax(-kl * 4.0, dim=1)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)
    print("Benchmark recorded.")
