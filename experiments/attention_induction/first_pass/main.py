"""Faithful hand-built induction head (first_pass).

NOTE: the original first_pass adapted a pretrained GPT-2 (vocab ~50k) and
returned an attention grid. That is fundamentally incompatible with this
goal's current contract, which requires next-token logits of shape
(batch, seq_len, vocab_size) == (64, 192, 128) over the synthetic 128-token
vocabulary. GPT-2's tokenizer/vocab cannot meet that contract, so we replace
it with a minimal, faithful induction-head circuit that implements exactly the
mechanism the goal probes: at a query position holding token t, find an earlier
position holding the same token t and copy the token that *followed* it.

All compute runs in torch on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


def induction_model_fn(vocab_size: int):
    """Return a model_fn(input_ids)->logits implementing an induction head on GPU."""

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        ids = torch.as_tensor(input_ids, dtype=torch.int64, device=DEVICE)  # (B, S)
        B, S = ids.shape

        # Previous-token map: token that followed each position (next token).
        # next_tok[b, p] = ids[b, p+1]; last position has no successor.
        next_tok = torch.full((B, S), -1, dtype=torch.int64, device=DEVICE)
        next_tok[:, :-1] = ids[:, 1:]

        # For each query position p, attend to earlier positions k < p whose
        # token matches ids[p]. The induction prediction is next_tok[k].
        # Build a (B, S, S) match matrix, strictly causal (k < p).
        same = ids.unsqueeze(2) == ids.unsqueeze(1)  # (B, S, S): same[b,p,k]
        idx = torch.arange(S, device=DEVICE)
        causal = idx.unsqueeze(1) > idx.unsqueeze(0)  # (S, S): p > k
        match = same & causal.unsqueeze(0)            # (B, S, S)

        # Prefer the most recent matching key (largest k). Give later keys
        # higher score so softmax (sharp) concentrates on the latest match.
        key_pos = idx.view(1, 1, S).to(torch.float32).expand(B, S, S)
        scores = torch.where(
            match,
            key_pos * 1.0e3,                       # sharp preference for latest match
            torch.full_like(key_pos, -1.0e9),
        )
        attn = torch.softmax(scores, dim=-1)        # (B, S, S)

        # Copy the successor token of the attended key into logits.
        # one_hot of next_tok over vocab, but next_tok==-1 -> zero row.
        valid_next = next_tok.clamp_min(0)          # (B, S)
        next_onehot = torch.zeros(B, S, vocab_size, dtype=torch.float32, device=DEVICE)
        next_onehot.scatter_(2, valid_next.unsqueeze(2), 1.0)
        next_onehot = next_onehot * (next_tok >= 0).unsqueeze(2).to(torch.float32)

        # logits[b, p, :] = sum_k attn[b,p,k] * next_onehot[b,k,:]
        copied = torch.bmm(attn, next_onehot)       # (B, S, vocab)

        # Scale into logit space so argmax/softmax are decisive but finite.
        logits = copied * 12.0
        return logits.detach().cpu().numpy().astype(np.float32)

    return model_fn


if __name__ == "__main__":
    task = load_task(__file__)
    model_fn = induction_model_fn(task.VOCAB_SIZE)
    payload = task.evaluate(model_fn)
    payload["model_name"] = "synthetic_induction"

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark recorded to {run_dir / 'benchmark.json'}")
