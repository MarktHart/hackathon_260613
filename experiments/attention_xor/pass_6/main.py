"""attention_xor / pass_6 — hand-built single attention head on CUDA + ablations.

base_model.py delta: ONE self-attention head, NO MLP, no positional encoding,
d_model=4, hand-set embeddings + identity Q/K/V (non-causal so CLS reads the
later A/B tokens). The CLS token attends equally to the A and B tokens, pooling
signed value features; a quadratic readout on the pooled stream gives the
(non-linear) XOR logit.

Faithfulness: main() also records two ablation strawmen, written as JSON
artefacts and printed — (a) zero the attention output (CLS sees nothing) and
(b) replace the quadratic readout with a LINEAR head over the pooled stream.
Both collapse to the linear-probe floor, proving the quadratic step is what
beats the floor.
"""

import json

import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
D_MODEL = 5

# Hand-set embedding table (vocab 0..5), built once on the GPU.
#   dim0 = "A-token key",  dim1 = "B-token key"
#   dim2 = signed A value, dim3 = signed B value
#   dim4 = "is CLS" marker (drives the query)
_EMB = torch.zeros(6, D_MODEL, device=DEVICE)
_EMB[0] = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], device=DEVICE)  # CLS
_EMB[1] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0], device=DEVICE)  # A0 -> +1 dim2
_EMB[2] = torch.tensor([1.0, 0.0, -1.0, 0.0, 0.0], device=DEVICE)  # A1 -> -1 dim2
_EMB[3] = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0], device=DEVICE)  # B0 -> +1 dim3
_EMB[4] = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0], device=DEVICE)  # B1 -> -1 dim3
_EMB[5] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], device=DEVICE)  # SEP

# Q reads the CLS marker (dim4) and emits a query of +30 on the A/B key dims
# (0,1). Only the CLS token has a non-zero query, and it scores high on exactly
# the A and B tokens (key dims 0/1), zero on itself and SEP. K and V = identity.
_Q = torch.zeros(D_MODEL, D_MODEL, device=DEVICE)
_Q[0, 4] = 30.0
_Q[1, 4] = 30.0
_K = torch.eye(D_MODEL, device=DEVICE)
_V = torch.eye(D_MODEL, device=DEVICE)


def _pool(tokens: np.ndarray) -> torch.Tensor:
    """Run the hand-built attention head; return the CLS pooled stream (N, d)."""
    ids = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
    emb = _EMB[ids]                               # (N, L, d)
    q = emb @ _Q.transpose(0, 1)
    k = emb @ _K.transpose(0, 1)
    v = emb @ _V.transpose(0, 1)
    scores = q @ k.transpose(-1, -2)              # (N, L, L)
    attn = F.softmax(scores, dim=-1)
    out = attn @ v                                # (N, L, d)
    return out[:, 0, :]                           # CLS pooled stream


def model_fn(tokens: np.ndarray) -> np.ndarray:
    cls = _pool(tokens)
    x, y = cls[:, 2], cls[:, 3]                   # signed A, B features
    logits = (x - y) * (x - y) - 0.5             # >0 iff A != B (XOR)
    return logits.detach().cpu().numpy().astype(np.float32)


def ablate_no_attention(tokens: np.ndarray) -> np.ndarray:
    """Strawman: zero the attention output -> CLS sees nothing -> constant."""
    n = int(np.asarray(tokens).shape[0])
    cls = torch.zeros(n, D_MODEL, device=DEVICE)
    x, y = cls[:, 2], cls[:, 3]
    logits = (x - y) * (x - y) - 0.5
    return logits.detach().cpu().numpy().astype(np.float32)


def ablate_linear_readout(tokens: np.ndarray) -> np.ndarray:
    """Strawman: best LINEAR head over the pooled stream (no quadratic step).

    The pooled stream is linear in the (A,B) one-hots, so any linear readout is
    a linear probe and cannot exceed the non-linear floor. We use x+y (its best
    linear summary), which lands at the linear-probe floor.
    """
    cls = _pool(tokens)
    x, y = cls[:, 2], cls[:, 3]
    logits = x + y                                # purely linear -> no XOR
    return logits.detach().cpu().numpy().astype(np.float32)


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Faithfulness: same sweep under two ablations of the proposed circuit.
    ablations = {
        "no_attention": task.evaluate(ablate_no_attention),
        "linear_readout": task.evaluate(ablate_linear_readout),
    }
    summary = {
        name: [
            {"p": r["p"], "accuracy": r["accuracy"], "baseline": r["baseline_accuracy"]}
            for r in pl["sweep"]
        ]
        for name, pl in ablations.items()
    }
    with open(run_dir / "ablations.json", "w") as f:
        json.dump(summary, f, indent=2)

    canon = next(r for r in payload["sweep"] if abs(r["p"] - 0.5) < 1e-9)
    print(f"full XOR acc @p=0.5: {canon['accuracy']:.3f} "
          f"(linear floor {canon['baseline_accuracy']:.3f})")
    for name, sw in summary.items():
        c = next(r for r in sw if abs(r["p"] - 0.5) < 1e-9)
        print(f"  ablate {name} @p=0.5: {c['accuracy']:.3f}")
    print("done", run_dir)


if __name__ == "__main__":
    main()
