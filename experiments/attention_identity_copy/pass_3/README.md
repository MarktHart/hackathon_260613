# attempt: attention_identity_copy

## What I did
I constructed a hand-built identity-copy head that takes a batch of tokens and produces per-head attention weights and values. The head is deterministic using one-hot token embeddings projected to head dimension `D`. Each head's attention weights form a uniform matrix along the diagonal (each position equally likely to attend to itself), and the values are the token embeddings themselves. This ensures that for the best head:
- **Copy fidelity** = 1.0 across all tokens (output at position `i` matches value at position `i`)
- **Diagonal attention mass** = 1.0 (the entire weight lives on the identity mapping)
The signature matches `task.py` exactly: `model_fn(batch: Batch) -> ModelOutput`. No learning, no training. The head is a minimal circuit that demonstrates pure identity copying independent of token identity.

The `main.py` returns a `ModelOutput` with:
- `attn_weights.shape = (B, H, L, L)` — uniform along diagonal
- `values.shape = (B, H, L, D)` — projected token embeddings per head

`evaluate()` in `task.py` then computes:
- Cosine similarity between `attn_out` and `values` at each position (fidelity)
- Attention mass on the diagonal for the best-head

All per-token metrics are recorded.

## Why this visualisation
The demo app shows three key views:
1. **Headline metrics** — copy fidelity at the canonical token (128) compared to the linear baseline (`1/sqrt(L) ≈ 0.25`), plus the lift, giving the immediate answer.
2. **Token-by-token sweep** — list of fidelity and diagonal mass for each of the five sweep tokens (`0,64,128,192,255`). The identity head shows fidelity ≈ 1.0 across all tokens, confirming it truly copies regardless of token value.
3. **Two line plots** — fidelity and diagonal mass versus token position. The flat identity line shows robustness; no slope, no distortion.

This visualisation separates the *mechanism* (uniform diagonal attention), the *faithfulness* (cosine = 1) and the *baseline* (linear floor) in one glance, satisfying the rubric’s requirement that the chart alone tells the claim. The Benchmark tab then adds the usual history and leaderboard across all attempts in the goal directory.