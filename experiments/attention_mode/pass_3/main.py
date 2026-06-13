"""Hand-built KL-based attention-mode classifier (pass_3).

Preserves the README intent: a per-head classifier that compares each head's
(L, L) attention matrix to the clean reference matrix for every mode, computes
row-averaged KL divergence, and softmaxes the negative divergences into a
probability over modes. No learning; the clean matrices are the very templates
`task.generate` uses. All arithmetic happens on CUDA.

Original bugs fixed against the current contract:
  - `assert isinstance(task, dataclass)` raised `TypeError` (`dataclass` is a
    decorator, not a type) — removed; `task` is a module.
  - the mode list comes from `task.MODES` (a tuple of the five canonical modes:
    positional, uniform, diagonal, induction, previous_token) and `task.CANONICAL_L`;
    there is no `task.modes`/`task.L`.
  - the softmax was `(-KL).log()` which takes log of negatives -> NaN. Softmax
    is now applied directly to the negative-KL scores.

Contract:
  input  : (n_heads, L, L) float32, rows sum to 1
  output : (n_heads, N_MODES) float32, rows sum to 1, in MODES order
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)
MODES = list(task.MODES)        # ("positional","uniform","diagonal","induction","previous_token")
N_MODES = len(MODES)
L = task.CANONICAL_L


def _positional_matrix(L, anchor=0):
    pat = np.zeros((L, L), dtype=np.float32)
    pat[:, anchor % L] = 1.0
    return pat


def _uniform_matrix(L):
    return np.full((L, L), 1.0 / L, dtype=np.float32)


def _diagonal_matrix(L):
    return np.eye(L, dtype=np.float32)


def _induction_matrix(L):
    pat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        pat[i, (i + 1) % L] = 1.0
    return pat


def _previous_token_matrix(L):
    pat = np.zeros((L, L), dtype=np.float32)
    for i in range(L):
        pat[i, (i - 1) % L] = 1.0
    return pat


_BUILDERS = {
    "positional": _positional_matrix,
    "uniform": _uniform_matrix,
    "diagonal": _diagonal_matrix,
    "induction": _induction_matrix,
    "previous_token": _previous_token_matrix,
}

# Stack clean templates in MODES order, normalise rows, and move to GPU.
_clean = np.stack([_BUILDERS[m](L) for m in MODES], axis=0)          # (M, L, L)
_clean = _clean / _clean.sum(axis=2, keepdims=True)
TEMPLATES = torch.as_tensor(_clean, dtype=torch.float32, device=DEVICE)  # (M, L, L)


def model_fn(attention_matrices: np.ndarray) -> np.ndarray:
    """Classify each head by min row-averaged KL to the clean templates (on GPU)."""
    pat = torch.as_tensor(attention_matrices, dtype=torch.float32, device=DEVICE)  # (H, L, L)
    H = pat.shape[0]
    eps = 1e-7

    p = pat.unsqueeze(1)                  # (H, 1, L, L)
    q = TEMPLATES.unsqueeze(0)           # (1, M, L, L)
    # KL(p || q) per query row: sum_k p * (log p - log q), then mean over rows.
    kl = (p * (torch.log(p + eps) - torch.log(q + eps))).sum(dim=3)   # (H, M, L)
    kl = kl.mean(dim=2)                                               # (H, M)

    # Lower KL = better match -> softmax over -KL (sharpened for decisiveness).
    probs = torch.softmax(-kl * 4.0, dim=1)                           # (H, M)
    probs = probs / probs.sum(dim=1, keepdim=True)
    return probs.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark recorded to {run_dir / 'benchmark.json'}")
