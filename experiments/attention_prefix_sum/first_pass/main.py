import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


def attention_head_with_positional_embeddings(input_ids: np.ndarray) -> np.ndarray:
    """A single causal attention head that computes the prefix sum mod vocab.

    Contract (from task.evaluate): input_ids is (B, L) int token ids in
    [0, vocab_size); return logits of shape (B, L, V).

    Mechanism (hand-built, no training):
      1. Build a causal attention pattern. Scores are (i+1) on causal keys
         {0..i} and -inf on future keys; row-wise softmax makes each row a
         uniform distribution over its causal prefix.
      2. The attended (mean) value at position i is mean(x[0..i]). Multiplying
         by the prefix length (i+1) recovers the cumulative sum x[0]+...+x[i].
      3. Take that mod vocab_size -> the target token id, and emit one-hot
         logits so argmax selects it.
    """
    vocab_size = 10  # canonical, from task.py

    ids = torch.as_tensor(input_ids, dtype=torch.float32, device=DEVICE)  # (B, L)
    B, L = ids.shape

    # Causal uniform attention via the positional index.
    pos = torch.arange(L, dtype=torch.float32, device=DEVICE).reshape(L, 1)  # (L,1)
    tril = torch.tril(torch.ones((L, L), dtype=torch.float32, device=DEVICE))
    causal_scores = tril * (pos + 1.0)                                       # (L, L)
    future_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=DEVICE), diagonal=1)
    causal_scores = causal_scores.masked_fill(future_mask, float("-inf"))
    causal_scores = causal_scores - causal_scores.max(dim=1, keepdim=True).values
    attn = torch.exp(causal_scores)
    attn = attn / attn.sum(dim=1, keepdim=True)                              # (L, L), uniform-causal

    # Attended mean value per query position, then rescale to the actual sum.
    mean_prefix = ids @ attn.t()                       # (B, L): row i = mean(x[0..i])
    prefix_len = (pos.reshape(1, L) + 1.0)             # (1, L) = i+1
    cumsum = mean_prefix * prefix_len                  # (B, L) cumulative sums
    target = torch.remainder(torch.round(cumsum), vocab_size).long()  # (B, L)
    target = target.clamp_(0, vocab_size - 1)

    logits = torch.zeros((B, L, vocab_size), dtype=torch.float32, device=DEVICE)
    logits.scatter_(-1, target.unsqueeze(-1), 1.0)
    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    task = load_task(__file__)
    payload = task.evaluate(attention_head_with_positional_embeddings)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Results saved to {run_dir}")
