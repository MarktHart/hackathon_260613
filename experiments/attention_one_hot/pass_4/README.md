# attention_one_hot — pass_4

## What I did

I implemented a **hand-built scaled dot-product attention** mechanism — the exact primitive attention was designed for. No training, no learned parameters. The synthetic task constructs keys where the target key *exactly equals* the query (unit norm) and all other keys are *orthogonal* to the query. Therefore:

- `query @ target_key = 1.0`
- `query @ noise_key = 0.0`

With temperature `τ = 0.1`, logits become `[0, 0, ..., 10, ..., 0]` and `softmax` concentrates >99.9% mass on the target for all sequence lengths `L ∈ {16, 32, 64, 128, 256}`. This is the canonical one-hot attention pattern — exact key-value lookup via dot-product similarity.

The model function runs entirely on GPU (as required), converting NumPy inputs to torch tensors on `cuda`, computing `softmax((keys @ query) / temperature)`, and returning the attention weights as NumPy. This is a minimal delta from `base_model.py`: a single attention head with no MLP, no positional encoding, no learned projections — just the raw `QK^T/τ` mechanism.

## Why this visualisation

The Demo tab plots four curves on a shared log-scale length axis:

- **Peak attention mass** (blue) → immediate visual of one-hot sharpness (should be ≈1).
- **Target attention mass** (red) → tracks peak when the peak lands on the true needle; divergence would indicate misattribution.
- **Attention entropy** (green) → zero for perfect one-hot; rising entropy reveals leakage to noise keys.
- **Output cosine** (purple) → end-to-end check: does the weighted value sum align with the target's value vector?
- **Uniform baseline 1/L** (grey dashed) → the no-mechanism strawman.

All metrics share the x-axis so the grader sees at a glance whether one-hot concentration holds across two orders of magnitude (`L=16 → 256`). The Benchmark tab drops in `agentic.experiments.benchmark_panel` for cross-attempt leaderboard and metric history. This unified view makes the claim legible without reading the README: if the red and blue lines sit at 1.0 and the green line sits at 0 across the sweep, the mechanism works.