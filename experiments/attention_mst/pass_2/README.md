# What I did
This is a **trained** attempt. The key observation is that Kruskal depends only
on the *ordering* of edge scores, so any per-edge monotone transform of the
noisy weights reproduces the strawman baseline exactly — denoising has to be
**structural** (an edge's score must depend on the rest of the graph). I train a
small permutation-equivariant self-attention network (a single block, the
smallest delta from `base_model.py`): each of the 12 heads is a token, the noisy
weight matrix is injected as an additive attention bias so nodes attend along
strong/low-weight edges, an MLP refines the node embeddings, and a symmetric
edge head reads `[h_i⊙h_j, |h_i−h_j|, w_ij]` to score every pair. It is trained
with BCE against planted-MST edge labels on graphs from the same generator using
seeds disjoint from the eval seed (42). The raw-weight skip lets the net fall
back to the baseline, so the structural attention can only help; the network
runs on CUDA in both training and inference inside `model_fn`. **Baseline** is
the no-mechanism `-noisy_weights` (computed internally by `task.evaluate`), and
as a **faithfulness / causal ablation** I re-evaluate the same trained net with
its attention output zeroed — knocking out the structural block collapses the
curve back toward the baseline, showing the recovery genuinely comes from
attention rather than the raw weights.

## Why this visualisation
The Demo tab plots MST edge-recovery **F1 vs noise level** for three series on
one axis: the trained denoiser (blue), the no-mechanism baseline (red), and the
attention-ablated net (green), with the canonical noise (0.5) marked. This is
the smallest artefact that carries the whole claim — the goal asks whether a
mechanism *denoises*, and denoising is exactly the blue-above-red gap sustained
across the noise sweep (the headline `mst_recovery` is the mean of the blue
curve). The green ablated curve falling to the baseline is the causal control in
the same frame, so a viewer can confirm in one glance that the lift is real and
comes from the attention block. The companion AUROC and weight-ratio panels show
ranking quality and tree-weight cost degrade gracefully rather than collapsing,
covering the operating-range question. The Benchmark tab carries the
cross-attempt leaderboard.
