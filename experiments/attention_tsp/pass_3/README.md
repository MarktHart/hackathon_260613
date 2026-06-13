# What I did

This is a **hand_built + interp** attempt. I take `base_model.py`'s single-head
`Attention` and make the smallest delta that solves NN routing: I replace the
learned token-embedding table with a fixed quadratic coordinate feature map
`φ(x,y) = [x, y, x²+y², 1]` and hand-set the 4×4 `W_q`, `W_k` so that the
ordinary dot-product attention score equals the **negative squared Euclidean
distance**: `Q_i·K_j = 2(x_i x_j + y_i y_j) − s_i − s_j = −‖c_i − c_j‖²` (with
`s = x²+y²`). The next-city argmax is therefore the nearest unvisited city —
the NN heuristic falls straight out of a standard attention head, not a bespoke
distance call. All compute runs in torch on CUDA; `W_q`/`W_k` are frozen
tensors (hardcoded-weights bonus). Crucially I add a **causal faithfulness
test**: ablating the key's `−s_j` feature collapses accuracy to ~random
(because the score loses its `‖k‖²` term and picks the farthest-in-direction
city), while ablating the query's `s_i` feature is provably inert (it only adds
a constant across candidates), pinning the work to exactly one feature.
`main.py` runs the full N=5..40 sweep plus both ablations and saves them.

# Why this visualisation

The Demo tab has two linked views that let a human check the claim directly.
**Left/right scatter:** the left panel colours every city by its raw `Q·K`
score and draws the argmax edge, so you see attention literally point at the
nearest city; the right panel plots `Q·K` against the *true* `−‖q−k‖²` and it
lands exactly on `y=x` for the full head — that line *is* the proof the head
computes negative distance, and it visibly bends away under either ablation.
**Grouped bar chart:** step-wise NN accuracy per problem size for full vs.
key-norm-ablation vs. query-norm-ablation vs. random baseline. The right
thing is on the y-axis (the benchmark's NN accuracy), the comparison is the
strawman the method must beat (random) plus the two ablations that isolate
*which feature* is load-bearing — so the chart shows not just that it works,
but why, and that removing the necessary feature breaks it while removing the
inert one does not.
