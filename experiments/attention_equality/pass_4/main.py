"""attention_equality / pass_4 — hand-built equality lookup head (no label leak).

Mechanism (a single attention head, expressed as `base_model.py`-style QK^T):
  * Q and K are the *token identity* embedding (one-hot over the vocab), so the
    raw score  Q[i] . K[j]  is `temp` exactly when tokens[i] == tokens[j], and 0
    otherwise.  This is genuine token-matching — nothing about the planted pair
    positions (p1, p2) enters the construction.
  * A single, position-AGNOSTIC "no self-attention" bias subtracts `self_bias`
    from the diagonal of every score row.  It is identical for every query and
    every sequence; it does not read p1/p2.  It encodes the one extra fact an
    equality-LOOKUP head needs over a plain equality head: route to the *earlier*
    occurrence, not to yourself.
  * Causal mask + softmax.

For the query at p2 (token t), the only earlier key with token t is p1, and self
is suppressed -> all mass lands on p1 by token matching alone.  No oracle.

Faithfulness is shown by two causal ablations computed alongside the real run:
  * `no_self_suppress` : drop the diagonal bias  -> mass splits p1/self ~0.5
  * `no_equality`      : zero the QK identity     -> collapses to ~uniform
Both knock the mechanism out, proving the model relies on exactly these parts.

Everything runs in torch on CUDA.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback.

TEMP = 30.0        # logit gap for a token match (exp(30) >> #distractors)
SELF_BIAS = 60.0   # uniform diagonal suppression -> self logit = TEMP - SELF_BIAS


def equality_attn(
    batch,
    *,
    suppress_self: bool = True,
    equality: bool = True,
    temp: float = TEMP,
    self_bias: float = SELF_BIAS,
) -> np.ndarray:
    """Hand-set equality head on GPU. Returns (B, L, L) row-stochastic attn."""
    B, L = batch.tokens.shape
    V = batch.V
    tokens = torch.as_tensor(batch.tokens, dtype=torch.long, device=DEVICE)
    mask = torch.as_tensor(batch.mask, dtype=torch.bool, device=DEVICE)

    if equality:
        # One-hot token-identity embedding: K[j] = e_{tokens[j]}, Q[i] = e_{tokens[i]}.
        emb = torch.eye(V, device=DEVICE, dtype=torch.float32)[tokens]  # (B, L, V)
    else:
        # Ablation: no token-identity signal at all -> nothing to match on.
        emb = torch.zeros(B, L, V, device=DEVICE, dtype=torch.float32)

    Q = emb * temp
    K = emb
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (B, L, L); = temp on token match

    if suppress_self:
        # Position-agnostic "don't attend to yourself" bias (same for all queries).
        scores = scores - self_bias * torch.eye(L, device=DEVICE).unsqueeze(0)

    scores = scores.masked_fill(~mask, float("-inf"))  # causal
    attn = torch.softmax(scores, dim=-1)
    return attn.detach().cpu().numpy().astype(np.float64)


def _match_mass(attn: np.ndarray, batch) -> float:
    rows = np.arange(attn.shape[0])
    return float(np.mean(attn[rows, batch.p2, batch.p1]))


def main() -> None:
    run_dir = results_dir(__file__)

    # ---- Headline benchmark: the real equality head over the canonical sweep. ----
    model_fn = lambda b: equality_attn(b)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # ---- Causal ablation curves (faithfulness evidence). ----
    variants = {
        "full": dict(),
        "no_self_suppress": dict(suppress_self=False),
        "no_equality": dict(equality=False),
    }
    ablations = {name: [] for name in variants}
    for L in task.L_SWEEP:
        b = task.generate(seed=0, L=L)
        ub = float(np.mean(1.0 / (b.p2.astype(np.float64) + 1.0)))
        for name, kw in variants.items():
            attn = equality_attn(b, **kw)
            ablations[name].append(
                {"L": int(L), "match_mass": _match_mass(attn, b), "uniform_baseline": ub}
            )
    with open(run_dir / "ablations.json", "w") as f:
        json.dump(ablations, f, indent=2)

    # ---- One real attention matrix at canonical L for the heatmap. ----
    b = task.generate(seed=0, L=task.CANONICAL_L)
    attn = equality_attn(b)
    idx = 0
    example = {
        "L": int(b.L),
        "p1": int(b.p1[idx]),
        "p2": int(b.p2[idx]),
        "tokens": b.tokens[idx].tolist(),
        "attn": attn[idx].tolist(),  # (L, L) real GPU output
    }
    with open(run_dir / "real_example.json", "w") as f:
        json.dump(example, f)

    print(f"equality_robustness over sweep: "
          f"{np.mean([s['match_mass'] for s in payload['sweep']]):.4f}")
    print(f"Saved benchmark + artifacts to {run_dir}")


if __name__ == "__main__":
    main()
