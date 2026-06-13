import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


def causal_uniform_head(input_ids: np.ndarray) -> np.ndarray:
    """Single causal uniform attention head computing prefix sum mod vocab.

    Contract (from task.evaluate): input_ids is (B, L) int token ids; return
    logits of shape (B, L, V).

    The head builds the ideal causal-uniform pattern by hand: scores are (i+1)
    on the causal prefix and -inf on future keys, so row-wise softmax yields a
    uniform distribution over {0..i}. The attended mean value times the prefix
    length (i+1) recovers the cumulative sum, which mod vocab_size gives the
    target token. One-hot logits make argmax pick that token. No learned
    parameters; runs on the GPU.
    """
    vocab_size = 10  # canonical, from task.py

    ids = torch.as_tensor(input_ids, dtype=torch.float32, device=DEVICE)  # (B, L)
    B, L = ids.shape

    pos = torch.arange(L, dtype=torch.float32, device=DEVICE).reshape(L, 1)
    tril = torch.tril(torch.ones((L, L), dtype=torch.float32, device=DEVICE))
    causal_scores = tril * (pos + 1.0)
    future_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=DEVICE), diagonal=1)
    causal_scores = causal_scores.masked_fill(future_mask, float("-inf"))
    causal_scores = causal_scores - causal_scores.max(dim=1, keepdim=True).values
    attn = torch.exp(causal_scores)
    attn = attn / attn.sum(dim=1, keepdim=True)        # (L, L) uniform-causal

    mean_prefix = ids @ attn.t()                       # (B, L) row i = mean(x[0..i])
    prefix_len = pos.reshape(1, L) + 1.0               # (1, L)
    cumsum = mean_prefix * prefix_len                  # (B, L) cumulative sums
    target = torch.remainder(torch.round(cumsum), vocab_size).long().clamp_(0, vocab_size - 1)

    logits = torch.zeros((B, L, vocab_size), dtype=torch.float32, device=DEVICE)
    logits.scatter_(-1, target.unsqueeze(-1), 1.0)
    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    task = load_task(__file__)
    payload = task.evaluate(causal_uniform_head)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Results saved to {run_dir}")
