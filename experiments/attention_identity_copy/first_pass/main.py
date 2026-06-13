"""
First-pass attempt at identity copying using scaled dot-product attention.

Hypothesis: a single attention head with scaled dot-product (QK^T / tau) and a
low temperature tau implements the copy primitive. With position-specific
query/key codes, the self-similarity <q_i, k_i> dominates, so a sharp softmax
routes position i's value back to position i — the identity copy.

The current task contract is `model_fn(batch: Batch) -> ModelOutput` returning
attn_weights (B, H, L, L) and values (B, H, L, D); this file is rewritten to
that contract while keeping the scaled-dot-product / low-temperature idea.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by pipeline; no CPU fallback

task = load_task(__file__)

# Canonical dims (must match task.py)
B = 32
L = 16
H = 8
D = 64

TEMPERATURE = 0.05  # tau — sharpens softmax onto the diagonal


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray  # shape (B, L), int32


@dataclass(frozen=True)
class ModelOutput:
    attn_weights: np.ndarray  # shape (B, H, L, L), float32
    values: np.ndarray        # shape (B, H, L, D), float32


def model_fn(batch: Batch) -> ModelOutput:
    B_, L_ = batch.tokens.shape
    assert B_ == B and L_ == L, \
        f"Batch dimensions must be ({B}, {L}), got ({B_}, {L_})"

    tokens = torch.as_tensor(batch.tokens, dtype=torch.int64, device=DEVICE)  # (B, L)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(0)

    # Unit-norm per-position query/key codes; <q_i, k_i> = 1 dominates, off
    # diagonal entries are ~0, so low-temperature softmax -> near one-hot diag.
    pos_code = torch.randn(H, L_, D, generator=g, device=DEVICE, dtype=torch.float32)
    pos_code = torch.nn.functional.normalize(pos_code, dim=-1)
    q = pos_code.view(1, H, L_, D).expand(B_, H, L_, D)
    k = pos_code.view(1, H, L_, D).expand(B_, H, L_, D)

    scores = torch.einsum("bhid,bhjd->bhij", q, k) / TEMPERATURE  # (B, H, L, L)
    attn_weights = torch.softmax(scores, dim=-1)                  # (B, H, L, L)

    # Token-dependent + positional value vectors so copy fidelity is meaningful.
    value_table = torch.randn(H, 256, D, generator=g, device=DEVICE, dtype=torch.float32)
    pos_vals = torch.randn(H, L_, D, generator=g, device=DEVICE, dtype=torch.float32)
    tok_vals = value_table[:, tokens, :].permute(1, 0, 2, 3)  # (B, H, L, D)
    values = tok_vals + pos_vals.view(1, H, L_, D)            # (B, H, L, D)

    return ModelOutput(
        attn_weights=attn_weights.detach().cpu().numpy().astype(np.float32),
        values=values.detach().cpu().numpy().astype(np.float32),
    )


def main():
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "payload.json").open("w") as f:
        json.dump(payload, f, indent=2)

    record_benchmark(__file__, run_dir, payload)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
