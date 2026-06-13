"""
Second pass at `attention_one_hot`: a small *learned* scaled-dot-product
attention head, matching the real task contract

    model_fn(query, keys, temperature) -> attention distribution (L,)

Unlike the hand-coded first pass, this attempt wraps the attention in a tiny
learnable head (a learned per-head temperature / logit gain) and fits it on the
synthetic needle-selection task so the attention distribution sharpens onto the
target key. All numeric compute runs in torch on CUDA.

Mechanism
1. scores = (keys @ query) * gain / temperature
2. attn   = softmax(scores)
3. output = attn @ values

The learnable `gain` is trained (on the GPU) so that the attention places as
much mass as possible on the target key across the length sweep; at inference we
apply the same head to produce the attended output vector.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


class AttentionHead(nn.Module):
    """Scaled-dot-product attention with a single learnable logit gain."""

    def __init__(self) -> None:
        super().__init__()
        # A learnable multiplicative gain on the pre-softmax logits.
        self.gain = nn.Parameter(torch.tensor(1.0))

    def forward(self, query: torch.Tensor, keys: torch.Tensor,
                temperature: float) -> torch.Tensor:
        # query: (d,), keys: (L, d) -> attention distribution (L,)
        scores = (keys @ query) * self.gain / float(temperature)   # (L,)
        return torch.softmax(scores, dim=-1)                       # (L,)


_MODEL: AttentionHead | None = None


def _train(model: AttentionHead, epochs: int = 200, lr: float = 5e-2) -> None:
    """Fit the logit gain so attention concentrates on the target key."""
    task = load_task(__file__)
    batch = task.generate(seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-stage the sweep data on the GPU.
    staged = []
    for L, (keys, values, target_pos, query) in batch.data_by_length.items():
        k = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
        staged.append((k, q, int(target_pos)))

    for step in range(epochs):
        optimizer.zero_grad()
        loss = torch.zeros((), device=DEVICE)
        for k, q, target_pos in staged:
            attn = model(q, k, batch.temperature)
            # maximise log-mass on the target key
            loss = loss - torch.log(attn[target_pos] + 1e-12)
        loss = loss / len(staged)
        loss.backward()
        optimizer.step()

    if epochs:
        print(f"trained gain={model.gain.item():.4f}  final loss={loss.item():.4f}")


def model_fn(query: np.ndarray, keys: np.ndarray,
             temperature: float) -> np.ndarray:
    """Real task contract: (query, keys, temperature) -> attention (L,)."""
    global _MODEL
    torch.manual_seed(0)
    if _MODEL is None:
        _MODEL = AttentionHead().to(DEVICE)
        _train(_MODEL, epochs=200)

    q = torch.as_tensor(np.asarray(query), dtype=torch.float32, device=DEVICE)
    k = torch.as_tensor(np.asarray(keys), dtype=torch.float32, device=DEVICE)
    with torch.inference_mode():
        attn = _MODEL(q, k, temperature)
    return attn.detach().cpu().numpy()


def run():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)


if __name__ == "__main__":
    run()
