"""First-pass attempt: a hand-built previous-token attention head on GPU.

The goal's contract is `model_fn(residual: (L, d) float32) -> (L, L) float32
logits`. `task.evaluate` applies its own causal mask + row-wise softmax and
measures how much attention mass lands on the previous token. We build a head
that reads the sinusoidal positional component of the residual stream and
produces logits L[i, j] that peak at j = i-1: query i is the position-(i)
embedding, key j is the position-(j) embedding shifted forward by one, so the
dot product is maximised when j = i-1. All compute runs on CUDA.
"""
from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

SEQ_LEN = 64
D = 64


def _positional_embeddings(seq_len: int, d: int) -> torch.Tensor:
    """Sinusoidal positional embeddings (seq_len, d) on the GPU.

    Matches the generator in task.py so we can align query i with key i-1.
    """
    pos = torch.arange(seq_len, dtype=torch.float64, device=DEVICE)[:, None]
    i = torch.arange(d, dtype=torch.float64, device=DEVICE)[None, :]
    div = torch.pow(torch.tensor(10000.0, dtype=torch.float64, device=DEVICE),
                    2.0 * torch.floor(i / 2.0) / d)
    angles = pos / div
    even = (torch.arange(d, device=DEVICE) % 2 == 0)[None, :]
    emb = torch.where(even, torch.sin(angles), torch.cos(angles))
    return emb.to(torch.float32)


def build_model_fn():
    """Return a model_fn matching the task signature: (L, d) -> (L, L) logits."""
    pos_emb = _positional_embeddings(SEQ_LEN, D)  # (L, d)
    # Shift positional embeddings forward by one: shifted[j] = pos_emb[j+1].
    # Then dot(residual_i, shifted[j]) ~ dot(pos_i, pos_{j+1}), maximised at
    # j+1 = i, i.e. j = i-1 -> a previous-token head.
    shifted = torch.zeros_like(pos_emb)
    shifted[:-1] = pos_emb[1:]

    def model_fn(residual: np.ndarray) -> np.ndarray:
        resid = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)  # (L, d)
        L = resid.shape[0]
        key = shifted[:L]  # (L, d)
        # Sharpen with a temperature so the previous-token mass is high.
        logits = (resid @ key.transpose(0, 1)) * 4.0  # (L, L)
        return logits.detach().cpu().numpy().astype(np.float32)

    return model_fn


def main() -> None:
    task = load_task(__file__)
    model_fn = build_model_fn()

    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    sweep = payload["sweep"]
    best = max(sweep, key=lambda r: r["prev_token_attention"])
    print(f"Done. Results in {run_dir}")
    print(f"Best prev-token attention: {best['prev_token_attention']:.4f} "
          f"at noise {best['noise']}")
    print(f"Uniform baseline: {best['uniform_baseline']:.4f}")


if __name__ == "__main__":
    main()
