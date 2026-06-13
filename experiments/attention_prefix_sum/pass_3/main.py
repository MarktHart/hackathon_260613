import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by pipeline


def _softmax(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable row-wise softmax over the last dim."""
    x = x - x.max(dim=-1, keepdim=True).values
    e = torch.exp(x)
    return e / e.sum(dim=-1, keepdim=True)


def causal_sum_head(batch_input_ids_t: torch.Tensor) -> np.ndarray:
    """Single-head attention circuit that builds the exact prefix-sum pattern
    and computes the prefix sum mod vocab_size.

    Returns logits of shape [B, L, V].

    The causal attention pattern: score[i, j] = (i+1) for j <= i and -inf for
    j > i. After softmax this is uniform over the causal prefix, so the attended
    mean value at position i is mean(x[0..i]); multiplying by (i+1) recovers the
    cumulative sum, and mod vocab_size gives the target token. One-hot logits
    select it. No learnable parameters.
    """
    B, L = batch_input_ids_t.shape
    vocab_size = 10  # hard-coded from task

    values = batch_input_ids_t.to(torch.float32)        # [B, L]

    pos = torch.arange(L, dtype=torch.float32, device=DEVICE).reshape(L, 1)  # [L,1]
    tril = torch.tril(torch.ones((L, L), dtype=torch.float32, device=DEVICE))
    causal_scores = tril * (pos + 1.0)                  # [L, L]
    future_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=DEVICE), diagonal=1)
    causal_scores = causal_scores.masked_fill(future_mask, float("-inf"))
    W = _softmax(causal_scores)                         # [L, L] uniform-causal

    # mean_prefix[b, i] = mean(x[b, 0..i])
    mean_prefix = values @ W.t()                        # [B, L]
    prefix_len = pos.reshape(1, L) + 1.0                # [1, L]
    cumsum = mean_prefix * prefix_len                   # [B, L] cumulative sum
    mod_cumsum = torch.remainder(torch.round(cumsum), vocab_size)
    target_idxs = mod_cumsum.long().clamp_(0, vocab_size - 1)  # [B, L]

    logits = torch.zeros((B, L, vocab_size), dtype=torch.float32, device=DEVICE)
    logits.scatter_(-1, target_idxs.unsqueeze(-1), 1.0)
    return logits.detach().cpu().numpy()


def _handbuilt_prefix_head(batch_input_ids: np.ndarray) -> np.ndarray:
    """Hand-built synthetic prefix-sum head: logits of shape [B, L, V]."""
    batch_input_ids_t = torch.as_tensor(batch_input_ids, dtype=torch.int64, device=DEVICE)
    return causal_sum_head(batch_input_ids_t)


if __name__ == "__main__":
    task = load_task(__file__)
    payload = task.evaluate(_handbuilt_prefix_head)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Results saved to {run_dir}")
