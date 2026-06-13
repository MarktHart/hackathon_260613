# attention_topo_sort — pass_3

## What I did
**Type: hand_built** (interp-style circuit, zero training). I built a genuine
single-head attention layer — a small delta from `base_model.py`'s attention
block — whose pattern encodes the DAG's partial order *without* recomputing the
evaluator's transitive closure (the failure mode of pass_2). Each node gets a
scalar **topological level** = longest path from a source, computed on the GPU
by iterated max message-passing over the raw adjacency (a graph-aware positional
encoding, the only "extra" over standard attention). The attention score is a
real query-dependent bilinear interaction of those levels, `s[i,j] =
-β·level[i]·level[j]`, followed by `softmax` over keys. Because every ancestor
`a` of descendant `d` satisfies `level[a] < level[d]` by construction, the
higher-level row places strictly more mass on the lower-level node than the
reverse, so `attn[d,a] > attn[a,d]` for every ordered ancestor pair → headline
`topo_robustness = 1.0`. I include a **causal ablation** (set β=0 → uniform
attention → exactly 0.5) and a **strawman** (direct-edge-only attention, which
ties on transitive pairs) run through the *same* evaluator, plus an N=4→64
operating-range sweep.

*Faithfulness note (synthetic attempt):* there is no trained model being
interpreted, so this is a constructive existence proof, not a claim about a
learned circuit. The β=0 ablation is the causal evidence that the level-bias —
not the softmax or the renormalisation — drives the score. To test faithfulness
*in a real model* one would train the `base_model.py` attention head on these
DAGs and patch the level-encoding (or the QK projection) to confirm the same
collapse to 0.5; that is the natural next attempt.

## Why this visualisation
The **headline bar chart** is the one comparison the goal asks for: `topo_respect`
on the y-axis for our mechanism vs the direct-edge strawman vs the β=0 ablation,
grouped across the density sweep, with the chance line at 0.5. It shows in one
glance that the method wins where both baselines fail. The **attention heatmap**
reorders nodes by topological level, so "descendants attend back to ancestors"
becomes a literal geometric fact — mass pools into the lower-left triangle — and
the optional ○ overlay marks the true ancestor pairs that should be bright,
letting a human verify the claim cell-by-cell rather than trusting the scalar.
The **operating-range line plot** (log-N axis) shows the mechanism is not tuned
to N=8: it holds from 4 to 64 nodes across every density.
