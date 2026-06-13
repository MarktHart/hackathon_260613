# What I did

This is a **hand_built** attempt that computes the exact transitive closure of
the graph using matrix powers on the GPU. The circuit is `(I + A)⁵ > 0`, where
`A` is the adjacency matrix and the exponent 5 covers the maximum path length
in the sweep (max diameter = 5). This is expressed as a sequence of torch
matrix multiplications on `cuda` — no training, no learned parameters. The
resulting affinity matrix is 1.0 for same-component pairs and 0.0 otherwise,
so pairwise F1 is 1.0 at every diameter. The adjacency baseline degrades as
diameter grows (it only sees 1-hop edges), so the lift is large and positive.

# Why this visualisation

The Demo tab shows per-diameter F1 for the model vs the adjacency baseline,
making the transitive-closure claim immediately legible: model F1 stays at 1.0
while baseline F1 drops from ~1.0 (diameter 1) to ~0.4 (diameter 5). The
Benchmark tab drops in the shared leaderboard so this hand-built oracle can be
compared against future learned or ablated attempts. The text readout is
deliberately minimal — the metric that matters is the gap between the two F1
curves, which the benchmark panel plots across the sweep.