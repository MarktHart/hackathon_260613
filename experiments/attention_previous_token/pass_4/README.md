# What I did

This is a **hand_built** previous-token attention head expressed as the minimal
delta from `base_model.py`'s single attention layer: one additive
**relative-position bias** on the logits, no MLP, one head. The bias is
`logits[i, j] = -alpha * ((i - j) - 1)^2` (T5 / ALiBi style), a smooth function
of the relative offset `i - j` peaked exactly at offset `1`, so every query `i`
attends to key `i - 1`. The head reads *position only* and never touches token
content, which is precisely what the goal asks for ("using positional
information while ignoring token content"). I chose this over a content-reading
sinusoidal-kernel QK head on purpose: because emb(i-1) and emb(i) are only
weakly separable, a content head caps near ~0.47 prev-token mass and collapses
under noise, whereas the position bias gives near-perfect clean mass *and*
robustness `= 1.0` (identical mass at every noise level, since content — the
only thing noise corrupts — is ignored). All real compute runs in torch on CUDA;
`task.evaluate` applies the causal mask and softmax.

**Baseline & faithfulness.** `main.py` measures two controls under identical
conditions. (1) *Baseline strawman*: a content-blind uniform head (zero logits)
lands at the `0.0594` uniform baseline, far below this head. (2) *Causal
ablation*: I sweep the bias **center** `c ∈ {0,1,2,3}` and record the attention
mass on each diagonal. The previous-token mass peaks sharply iff `c = 1`; at
`c = 0` the mass moves to self, at `c = 2` to two-back — moving the cause moves
the effect, which is the synthetic analogue of an activation patch and proves
the `(i-j)=1` term *is* the mechanism rather than a coincidence.

# Why this visualisation

The Demo tab leads with the **attention heatmap** on the clean sequence: a
correct previous-token head is a single bright band one cell below the diagonal,
true-or-false at a glance. Beneath it the **robustness line plot** puts the three
diagonals (prev `i-1`, self `i`, two-back `i-2`) and the uniform baseline on one
axis against noise — the right grain, because the goal asks both "does it hit
`i-1`?" (prev line high) and "does it stay there under noise?" (flat line,
robustness 1.0). The **ablation bar chart** is the causal panel: as the bias
center slides 0→3, the tallest bar walks from self → prev → two-back,
visually pinning the mechanism to offset 1. The Benchmark tab drops in the
shared leaderboard for comparison on `prev_token_attn_canonical`.
