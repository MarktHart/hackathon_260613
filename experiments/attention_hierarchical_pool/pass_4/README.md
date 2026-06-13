# attention_hierarchical_pool / pass_4

## What I did

This is a **hand_built** attempt. Each of the 96 heads is one real `Q@Kᵀ`
attention head whose score matrix is produced by **hand-set Q/K projections** on
fixed positional features `f(p) = [p, p², 1]`: with `q_i = [i/σ², −1/(2σ²)]` and
`k_j = [j, j²]` the dot product collapses to `−(i−j)²/(2σ²) + const_i`, so the
softmax row is an unmasked Gaussian over keys with std σ — a genuine attention
pattern computed in torch on CUDA, not a painted indicator matrix. The single
delta from `base_model.py` is one attention layer (no MLP, no residual, no
learning) with a depth-indexed width: **σ is the only thing that changes with
depth**, growing geometrically from `0.6` (layer 0) to `16` (layer 11).

The decisive change from the prior attempt (pass_3) is that **nothing is
masked.** pass_3 hard-masked attention to the query's own chunk, which made
`superchunk_concentration` trivially `1.0` and never demonstrated the
chunk → super-chunk pooling the goal explicitly asks for. Here the Gaussian is
free to spill across boundaries, so all three concentrations are *genuinely
measured*: as depth rises the receptive field sweeps through every level of the
tree — early layers concentrate on the token (high local), mid layers fill the
16-token chunk, and late layers pull mass **out of the chunk** (chunk_conc drops
to ≈ 0.38) while it stays **inside the 64-token super-chunk** (super-chunk_conc
stays ≈ 0.95). That redistribution *is* coarse pooling, and every transition is
exhibited rather than hard-coded. Measured headline
`hierarchical_robustness_canonical ≈ 2` (> 1 ⇒ fine→coarse shift) against the
uniform-within-chunk strawman (`≈ 1.0`). A **faithfulness ablation is actually
run and saved** as `results/<run>/ablation.json`: freezing σ at its geometric
mean for every layer removes the only causal knob and collapses the robustness
to ≈ 1.0, confirming the depth schedule — and nothing else — produces the
hierarchy.

## Why this visualisation

The Demo tab puts the claim on three coordinated panels. The **left depth curve**
plots all three concentrations (local, chunk, super-chunk) on one axis against
layer: you watch `local` collapse first, then `chunk` fall while `super-chunk`
stays pinned near the top — the literal picture of mass walking up the tree from
token to chunk to super-chunk, which is exactly what the goal asks. The **right
curve** distils that into the headline `spread = chunk/local` rising with depth,
with the flat uniform / flat-σ line at ≈ 1.0 for reference and the resulting
robustness printed so the number and the picture agree. The **ablation bar chart**
reads the saved `ablation.json` and stands the canonical robustness next to the
flat-σ knockout and the uniform baseline, making the causal claim checkable at a
glance. The **per-head heatmap** (sliders for layer/head) shows the first
super-chunk with cyan chunk borders, so you can watch a single head's Gaussian
go from a bright diagonal dot (early) to a broad block that *crosses* the chunk
borders but stays within the super-chunk (late). The Benchmark tab drops in
`benchmark_panel` for the cross-attempt leaderboard and metric history.
