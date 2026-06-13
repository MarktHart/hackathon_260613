# attention_bipartite · pass_2

## What I did
**Type: hand_built circuit + a learned strawman (faithfulness/generalisation test).**
This is a single-attention-layer delta from `base_model.py`: I keep scaled
dot-product attention and add **one fixed additive mask** to the scores before
softmax — `score(i, j) = -inf` whenever query `i` and key `j` are in the *same*
group (exactly the shape of `base_model.Attention`'s causal mask, but bipartite).
That one line forces every query to attend only to the *other* group, and it is
the entire mechanism — ablating it (mask off) recovers the failing baseline. All
compute runs in torch on CUDA. The benchmark payload follows the goal README's
documented contract, but I re-derive `mean_attn_within/between` and
`retrieval_acc` from the attention weights the model **actually** produces and
with a working softmax — the shipped `task.evaluate` calls the nonexistent
`np.softmax` and scores from raw q/k (ignoring `model_fn`), so it can neither run
nor reflect any mechanism. Key results (canonical, num_heads=4): bipartite score
**0.125** (mask) vs **−0.001** (baseline); retrieval **0.49** vs **0.21** against
a **0.5** ceiling (two same-feature keys per group). The decisive faithfulness
check: tokens carry *content only* (`q=k=v=feature_base+noise`), so the
within-group vs cross-group target differ only by **position** — a learned
content-only `W_q/W_k` overfits one batch (train 0.49) but scores **0.00** on a
held-out seed, while the positional mask transfers (held-out **0.52**).

## Why this visualisation
Three stacked views, each killing one alternative explanation. (1) **Heatmaps**
(baseline vs mask, group boundary drawn in white) show the qualitative claim at a
glance: the baseline lights up the diagonal/within-group blocks, the mask lights
up only the off-diagonal cross-group blocks — the literal definition of bipartite
attention. (2) **Bipartite-score bars** across the `num_heads` sweep put the
headline metric (between − within) on the y-axis with the no-mask strawman beside
it, so "ours works *while* the baseline doesn't" is one comparison, and the flat
bars show head-count robustness. (3) **Generalisation curve** plots retrieval vs
training step for the content-only model — train high, held-out pinned at 0 —
with the mask's held-out accuracy as a dashed line, making "the model would have
to use *position*, and only the mask does" visible rather than asserted.
