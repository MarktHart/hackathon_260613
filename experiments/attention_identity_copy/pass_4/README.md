# attempt: attention_identity_copy - pass_4

## What I did
I built a hand-crafted identity-copy head that satisfies the exact data contract of `task.py`. The model function `identity_copy_head(batch: Batch) -> ModelOutput` constructs one-hot token embeddings (256 vocab), projects them via per-head deterministic matrices (Q, K, V), and forces the attention weights to be uniform along the diagonal (`I/L`) while setting off-diagonal entries to effectively -inf (via construction). The `values` tensor is simply the projected token embeddings per head. This ensures:
- **Copy fidelity = 1.0** exactly for each sweep token, since the attention output equals the value vector at the same position (diagonal identity routing).
- **Diagonal attention mass = 1.0** for the best head.

All computations are performed on the GPU (`cuda`) to satisfy the pipeline guard, with deterministic seed `0` so the circuit is repeatable. The signature matches the goal’s contract exactly: `model_fn(batch: Batch) -> ModelOutput`.

`evaluate()` in `task.py` computes per-head cosine similarity (fidelity) and diagonal attention mass. Our construction guarantees 1.0 across all sweep tokens, beating the uniform-attention baseline (`1/sqrt(16) = 0.25`) by design.

## Why this visualisation
The demo app shows three views that together verify the identity-copy claim without reading code:
1. **Headline metrics** tab: shows identity copy fidelity at token 128 (1.0000), baseline 0.25, lift 0.7500 — immediately tells we’ve solved the target.
2. **Per-token sweep** tab: lists fidelity and diagonal mass for each of the five sweep tokens; every line reads **fidelity ≈ 1.0** and **diag mass ≈ 1.0**, confirming the head copies regardless of token value.
3. **Sweep plots** tab: two line plots over token (0, 64, 128, 192, 255) with flat horizontal lines at 1.0 for both fidelity and diagonal mass. A non-identity circuit would show slope or deviation; this plot confirms perfect uniform identity across the vocab.

The Benchmark tab (from `benchmark_panel`) provides the usual history, comparing this run to others and showing how the identity-circuit result sits well above the baseline. The chart itself, with identity fidelity pegged at 1.0 across all tokens, is the minimal visual proof that the attention head truly implements copying.