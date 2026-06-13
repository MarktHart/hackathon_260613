# What I did

This is a **hand_built attention circuit** (interp-style, no training) — a
deliberate departure from `first_pass`, which used raw `(I+A)^5` matrix powers
and answered the goal's *attention* question by avoiding attention entirely.
Here the transitive closure is expressed as the smallest delta from
`experiments/base_model.py`: I keep its `Attention` block but replace the
learned Q/K with an **adjacency-derived additive attention bias** (a graph
relative-position bias — score 0 for self/neighbour, −∞ otherwise, so softmax
gives uniform attention over the 1-hop neighbourhood), use **identity one-hot
token embeddings** as the values, add a hard-threshold read-out per layer, drop
the MLP, and stack `depth` layers (no causal mask, no RoPE). One attention
layer propagates exactly one hop, so after `L` layers a node's residual stream
marks every node within `L` hops; for `L ≥ diameter` that is the full connected
component, giving exact same-component affinity (`transitive_closure_robustness
= 1.0`, full positive lift over the adjacency baseline at every diameter > 1).
The benchmark payload uses `depth = N`, which guarantees `depth ≥ diameter` for
**any** graph size — fixing `first_pass`'s hardcoded exponent-5 brittleness.

**Faithfulness / causal evidence.** The attempt is synthetic (no trained model
to patch), so the causal handle is the mechanism's own depth knob, recorded in
`ablation.json`: at **depth 1** the read-out is provably identical to the 1-hop
adjacency baseline, and the closure **breaks the instant `depth < diameter`** —
removing attention hops collapses the answer back to the strawman, which is
exactly the ablation a faithfulness check on a learned model would perform. A
trained-model version would set Q/K so attention scores reproduce this
adjacency mask, then ablate the per-layer attention writes and watch F1 fall to
the baseline at each depth.

# Why this visualisation

The Demo tab is built around the one claim that distinguishes closure from
adjacency: **does extra attention depth buy reach?** Panel (A) plots pairwise
F1 against attention depth with one line per diameter and the adjacency
baseline overlaid — the model line starts *on* the baseline at depth 1 and
steps up to 1.0 exactly at `depth == diameter`, so the "needs ≥ diameter hops"
mechanism and the strawman gap are both legible in a single chart. Panel (B)
makes it concrete: pick a diameter, slide the depth, and watch the predicted
same-component heatmap fill in hop-by-hop until it matches ground truth, with
cells colour-coded green (correct merge) / orange (missed, needs more hops) /
red (false merge) against the reordered component blocks. Together they show
*where* the circuit succeeds and *where* it would break, rather than asserting
a single F1 number.
