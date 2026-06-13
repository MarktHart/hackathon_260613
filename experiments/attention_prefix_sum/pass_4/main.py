"""attention_prefix_sum / pass_4 — "Clock" readout for prefix-sum-mod-V.

Hand-built single-layer, single-head attention circuit. The DIFFERENT idea
from earlier passes: the modular reduction is NOT done by a `remainder` call.
Instead it is performed by a fixed **Fourier ("clock") unembedding** — exactly
the representation that real transformers grok for modular arithmetic
(Nanda et al. 2023). Attention does the linear accumulation; the periodic
readout does the `mod V`. Everything runs in torch on CUDA.

Mechanism (delta from base_model.py):
  1. token embedding = identity (the scalar token value is the value stream).
  2. ONE attention head with hand-set causal-uniform weights
     W[i,j] = 1/(i+1) for j<=i  (the triangular prefix mask).
     This yields mean_prefix[i] = mean(x[0..i]).
  3. position-conditioned scale by (i+1) recovers the exact cumulative sum
     S[i] = sum(x[0..i]).  (softmax attention can only average, so the count
     multiply is the standard way to turn a mean into a sum.)
  4. fixed Fourier unembedding turns S into class logits:
        logit_c = sum_f cos(2*pi*f*(S-c)/V) = V * 1[S == c (mod V)]
     -> argmax picks (S mod V) exactly, for ANY length. No MLP.

We also run two ABLATIONS through the identical task.evaluate to make the
faithfulness/baseline story checkable:
  * no_attention   : self-only attention (no prefix mask) -> predicts x_t.
  * magnitude_read : correct accumulation but a NON-periodic readout that just
                     reads S as the class -> only right while S < V, so it
                     decays with length. Isolates the work the clock does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback
V = 10           # vocab size (from task.py)
PI = math.pi


# --------------------------------------------------------------------------- #
# Circuit pieces (all on GPU)
# --------------------------------------------------------------------------- #
def _accumulate(x: torch.Tensor, mode: str) -> torch.Tensor:
    """x: [B, L] float -> S: [B, L] cumulative sum (rounded to int)."""
    B, L = x.shape
    pos = torch.arange(L, device=DEVICE, dtype=torch.float32)
    if mode == "causal":
        # triangular prefix mask, row-normalised -> uniform over the prefix
        tril = torch.tril(torch.ones(L, L, device=DEVICE, dtype=torch.float32))
        w_attn = tril / tril.sum(dim=-1, keepdim=True)        # W[i,j] = 1/(i+1)
        mean_prefix = x @ w_attn.t()                          # [B, L]
        S = mean_prefix * (pos + 1.0)                         # mean -> exact sum
    elif mode == "self":
        S = x.clone()                                         # only attend to self
    else:
        raise ValueError(mode)
    # S is a sum of integers; round() just cleans float32 noise (not a `mod`).
    return torch.round(S)


def _readout(S: torch.Tensor, mode: str) -> torch.Tensor:
    """S: [B, L] -> logits: [B, L, V]."""
    B, L = S.shape
    Sf = S.reshape(-1)                                        # [N]
    if mode == "fourier":
        f = torch.arange(V, device=DEVICE, dtype=torch.float32)   # frequencies 0..V-1
        c = torch.arange(V, device=DEVICE, dtype=torch.float32)   # classes 0..V-1
        ang = 2.0 * PI * torch.outer(Sf, f) / V              # [N, V]
        a, b = torch.cos(ang), torch.sin(ang)
        w_cos = torch.cos(2.0 * PI * torch.outer(f, c) / V)  # [F, V]
        w_sin = torch.sin(2.0 * PI * torch.outer(f, c) / V)  # [F, V]
        logits = a @ w_cos + b @ w_sin                       # = V * delta(S mod V, c)
    elif mode == "magnitude":
        # non-periodic strawman: pick the class whose value is closest to S
        c = torch.arange(V, device=DEVICE, dtype=torch.float32)
        logits = -((c[None, :] - Sf[:, None]) ** 2)          # argmax = clamp(round(S))
    else:
        raise ValueError(mode)
    return logits.reshape(B, L, V)


def make_fn(attn_mode: str, readout_mode: str):
    """Return a task-compatible model_fn: input_ids[B,L] -> logits[B,L,V]."""
    def fn(input_ids: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(input_ids, dtype=torch.float32, device=DEVICE)
        S = _accumulate(x, attn_mode)
        logits = _readout(S, readout_mode)
        return logits.detach().cpu().numpy().astype(np.float32)
    return fn


full_fn = make_fn("causal", "fourier")          # the proposed mechanism
no_attention_fn = make_fn("self", "fourier")    # ablation: no prefix mask
magnitude_fn = make_fn("causal", "magnitude")   # ablation: no clock readout


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def _sweep_acc(payload: dict) -> dict[int, float]:
    return {r["seq_len"]: (r["correct"] / r["total"] if r["total"] else 0.0)
            for r in payload["sweep"]}


if __name__ == "__main__":
    task = load_task(__file__)
    run_dir = Path(results_dir(__file__))

    # Headline: the full clock circuit.
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)

    # Ablations through the SAME evaluator (for the demo's causal evidence).
    ablations = {
        "full (causal + clock)": _sweep_acc(payload),
        "no_attention (self + clock)": _sweep_acc(task.evaluate(no_attention_fn)),
        "magnitude readout (no mod)": _sweep_acc(task.evaluate(magnitude_fn)),
        "random baseline": {L: 1.0 / V for L in [4, 8, 16, 32, 64]},
    }
    (run_dir / "ablations.json").write_text(json.dumps(ablations, indent=2))

    # Save the L=16 attention pattern + the clock readout for visualisation.
    L = 16
    tril = torch.tril(torch.ones(L, L, device=DEVICE, dtype=torch.float32))
    w_attn = (tril / tril.sum(dim=-1, keepdim=True)).detach().cpu().numpy()
    np.save(run_dir / "attention_L16.npy", w_attn)

    f = np.arange(V)
    c = np.arange(V)
    w_cos = np.cos(2 * np.pi * np.outer(f, c) / V)
    np.save(run_dir / "clock_wcos.npy", w_cos.astype(np.float32))

    print(f"prefix_acc_canonical = {_sweep_acc(payload)[16]:.4f}")
    print(f"Results saved to {run_dir}")
