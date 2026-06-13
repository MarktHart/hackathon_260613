"""
Attention deduplication — hand-built duplicate-token head.

Approach (hand_built): no training. We construct a single causal attention
layer whose score matrix implements the duplicate-token motif analytically:

    score[s, q, k] =  BIG + REC * k      if k < q and tok_k == tok_q   (earlier match)
                      SELF               if k == q                     (diagonal)
                      -inf               otherwise                     (non-causal / non-match)

A softmax over k then routes a *duplicate* query almost entirely onto the
**most recent earlier position holding the same token** (largest k among the
matches wins because of the `REC * k` recency ramp, and BIG >> SELF makes any
real match beat the diagonal). A *first-seen* query has no earlier match, so
only the diagonal survives and the token stays on itself — exactly what a clean
dedupe head should do.

This is `base_model.py` reduced to a single attention layer with no MLP, and
with the QK score replaced by an exact token-identity + recency rule instead of
a learned bilinear form. All compute runs on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback

# Hand-set "weights" of the circuit.
BIG = 30.0   # match bonus: any earlier same-token key dominates the diagonal
REC = 4.0    # recency ramp: among earlier matches, larger index (more recent) wins
SELF = 0.0   # diagonal logit: where first-seen tokens park their mass
NEG = -1e9   # masked-out logit (non-causal or earlier non-match)


def make_model_fn():
    """Return the hand-built duplicate-token-head attention function."""

    def model_fn(tokens: np.ndarray) -> np.ndarray:
        toks = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
        n, L = toks.shape

        q_idx = torch.arange(L, device=DEVICE).view(1, L, 1)   # query position
        k_idx = torch.arange(L, device=DEVICE).view(1, 1, L)   # key position

        match = toks[:, :, None] == toks[:, None, :]           # (N,L,L) tok_q==tok_k
        earlier = k_idx < q_idx
        diag = k_idx == q_idx

        scores = torch.full((n, L, L), NEG, dtype=torch.float32, device=DEVICE)
        # earlier same-token keys: match bonus + recency ramp
        ramp = BIG + REC * k_idx.to(torch.float32)             # (1,1,L)
        scores = torch.where(earlier & match, ramp.expand(n, L, L), scores)
        # diagonal: fixed self logit (first-seen tokens stay home)
        scores = torch.where(diag.expand(n, L, L),
                             torch.full_like(scores, SELF), scores)

        attn = torch.softmax(scores, dim=-1)                   # row-stochastic over keys
        return attn.detach().cpu().numpy().astype(np.float64)

    return model_fn


def main() -> None:
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    model_fn = make_model_fn()
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # Save the canonical-condition attention example for the Demo tab.
    batch = task.generate(0)
    canon = next(s for s in batch.slices
                 if abs(s["dup_rate"] - batch.canonical_dup_rate) < 1e-9)
    attn = model_fn(canon["tokens"])
    np.savez(
        run_dir / "example.npz",
        tokens=canon["tokens"],
        prev=canon["prev"],
        attn=attn.astype(np.float32),
        dup_rate=np.float64(canon["dup_rate"]),
    )

    # Also keep the raw payload for the sweep bar chart in the demo.
    import json
    (run_dir / "payload.json").write_text(json.dumps(payload, indent=2))

    sw = {r["dup_rate"]: round(r["dedup_mass"], 4) for r in payload["sweep"]}
    print("dedup_mass by dup_rate:", sw)
    print("artefacts ->", run_dir)


if __name__ == "__main__":
    main()
