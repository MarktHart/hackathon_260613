# pass_3 — hand-built QK/softmax stack-matching circuit

## What I did
This is a **hand_built** attempt, but — unlike the prior oracle that *wrote*
the answer straight into the attention tensor — here the stack-matching pattern
**emerges from a genuine attention computation** (`scores = Q·Kᵀ → causal mask →
softmax`). It is the smallest delta from `base_model.py`'s `Attention` layer: a
single head (replicated over 4), **no MLP**, with the `qkv` projection *hand-set*
instead of learned. A depth-counter sublayer computes each token's match-depth
`m` from a causal cumsum of `(+1` `−1)`; a matching pair provably shares the same
`m`. The Q/K features then score every (close `i`, key `j`) pair as
`A·1[m_i=m_j] + W·is_open_j + C·pos_j` with strictly tiered magnitudes
(`A=1e4 ≫ W=1e3 ≫ C·pos`), so the real softmax resolves them lexicographically:
keep same-depth keys, prefer openings, break ties by recency — which is provably
the true match. Result: `mean_attn_to_match ≈ 0.99` at every depth (vs ≈0.07
chance), and it is depth-invariant by construction.

**Faithfulness / causal check.** Because this is a hand-built circuit, the
faithfulness test is an *ablation of the circuit itself*: zeroing the depth term
(`A=0`) leaves a recency-only head that attends to the nearest opening. The app's
sweep shows this collapses as nesting deepens while the full circuit stays flat —
direct causal evidence that the **depth feature is what produces the matching**,
not recency or the is-open gate alone. For a *trained* model the analogous check
would be activation-patching the depth-counter subspace and watching match-attention
break; that is the natural next attempt.

## Why this visualisation
Two views, each checking a different half of the claim. The **heatmap** proves
the mechanism is real per-token: cyan boxes mark the ground-truth matching pairs,
so a single bright pixel inside every `)`-row's box is the visual signature of
"close attends to its open" — and it's the post-softmax weight, not a planted
value. The **depth-sweep line plot** is the scientific argument: y-axis is mean
attention on the matching `(` (the exact headline metric), x-axis is nesting
depth (the goal's sweep axis), and three lines — full circuit, depth-ablated
recency-only, and uniform chance — put the success *and* its strawman on the same
axes. The gap that opens up with depth between the green and red lines is the
whole point: stack tracking, not proximity, is doing the work.
