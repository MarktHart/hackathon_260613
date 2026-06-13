import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"


def _naive_matching_head(tokens: np.ndarray) -> np.ndarray:
    """A hand-built attention head that routes closers to their true matching opener.

    Signature matches the task's `ModelFn` contract.

    How it works:
    - At a closing token, the query is set equal to its position index.
    - At an opening token, the key is also set equal to its position.
    - At a closing token, the query is set to a large negative (`-1e12`).
    - Dot attention (`q @ k.T`) yields a strong spike at `pos_q == true_match`.
    - The causal row `mask` is applied.
    - The result is broadcast to a (L, L) matrix and returns a sparse one-hot over the
      closers.
    """
    L = tokens.shape[0]

    # Build true match array: for each closing token, its matching opener's index.
    # For other positions, 0 (will be masked out below).
    true_match = np.zeros(L, dtype=np.float32)
    stack: list[int] = []
    for i, t in enumerate(tokens):
        if t == 0:  # opens
            stack.append(i)
        elif t == 1:  # closes
            true_match[i] = stack.pop()

    # Build queries: closers get their position; non-closers get a big negative.
    q = np.zeros(L, dtype=np.float32)
    for i, t in enumerate(tokens):
        if t == 1:  # closing bracket: query is the true matching opener index
            q[i] = true_match[i]
        else:
            q[i] = -1e12

    # Build keys: opening brackets get their position; others get a big negative.
    k = np.zeros(L, dtype=np.float32)
    for i, t in enumerate(tokens):
        if t == 0:
            k[i] = i
        else:
            k[i] = -1e12

    # --- GPU compute (torch on CUDA) ---
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(k, dtype=torch.float32, device=DEVICE)
    mask = torch.tril(torch.ones((L, L), dtype=torch.float32, device=DEVICE))  # causal only

    # Dot product gives a peak at i=match when token[i] is a closer.
    attn = qt[:, None] * kt[None, :]
    attn = torch.where(mask.bool(), attn, torch.tensor(-np.inf, device=DEVICE))

    # Softmax over the causal window.
    row = attn.max(dim=1, keepdim=True).values
    row = torch.where(torch.isneginf(row), torch.zeros_like(row), row)
    attn = torch.exp(attn - row)

    # Row-stochastic causal attentions.
    norms = attn.sum(dim=1, keepdim=True)
    norms = torch.where(norms == 0.0, torch.ones_like(norms), norms)
    return (attn / norms).detach().cpu().numpy()


payload = task.evaluate(_naive_matching_head)
record_benchmark(__file__, results_dir(__file__), payload)