# attention_hierarchical_pool / pass_3

## What I did

This is a **hand_built** attempt. Each of the 96 heads is a single, real QK
attention head whose score matrix is produced by **hand-set Q/K projection
weights** acting on fixed positional features `f(p) = [p, p², 1]`. With
`q_i = [i/σ², -1/(2σ²)]` and `k_j = [j, j²]` the dot product collapses to
`q_i·k_j = -(i-j)²/(2σ²) + const_i`, so softmax over keys is a Gaussian centred
on the query with standard deviation σ — a genuine attention pattern, computed
as `Q @ Kᵀ` in torch on CUDA, not a painted indicator matrix. Cross-chunk keys
are masked so pooling respects the chunk boundary. **The only quantity that
changes with depth is σ**: it grows geometrically from `0.55` (layer 0) to
`7.0` (layer 11), so early heads pool a tight ±-token neighbourhood (fine) and
late heads pool the whole 16-token chunk (coarse). This is the smallest
delta from `base_model.py` that expresses the hierarchy: one attention layer,
no MLP, no residual, no learning — just a depth-indexed width on a hand-set
positional head. Measured headline `hierarchical_robustness_canonical = 1.80`
(> 1 ⇒ depth-dependent fine→coarse shift); the per-layer spread rises
monotonically while the uniform-within-chunk strawman is depth-invariant and
scores ≈ 1.0. A faithfulness check is built in by construction: σ is the sole
causal knob, so flattening the σ-schedule (constant width) collapses the
robustness to 1.0 — knock out the depth dependence and the mechanism dies.

## Why this visualisation

The Demo tab puts the claim on two axes that map directly onto the benchmark.
The **left depth curve** plots `spread = chunk_conc / local_conc` against layer
— exactly the quantity the headline metric ratios between late and early layers
— and overlays the flat uniform-within-chunk baseline; the *rise* of the red
curve above that flat grey line is the hierarchy, and the title prints the
resulting robustness so the number and the picture agree. The **right curve**
shows *why* it rises: local mass collapses with depth while chunk mass stays
≈ 1, i.e. mass spreads outward but never leaves the chunk. The **per-head
heatmap** (sliders for layer/head) shows three interior chunks with cyan chunk
borders, so you can watch a single head's Gaussian widen from a bright diagonal
dot (early) to a broad within-chunk block (late) and confirm it never crosses a
border. The Benchmark tab drops in `benchmark_panel` for the cross-attempt
leaderboard and metric history.
