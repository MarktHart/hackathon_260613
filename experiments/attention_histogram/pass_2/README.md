# attention_histogram / pass_2

## What I did

This is a **hand_built** attempt (no training; knobs β0=8, β=16, n_iter=2 are
hand-set). The delta from `base_model.py` is an **iterative query-refinement**
attention head: instead of scoring the keys against the noisy query once, it
first runs a soft attention pass `a = softmax(β0·Kq)`, replaces the query with
the resulting key-weighted average `q ← unit(Kᵀa)`, and repeats before the
final scoring `β·Kq`. The refined query is a convex combination of the keys
dominated by the target, so it is a cleaner estimate of the target direction
than the noise-corrupted query — this is the attention power-iteration the
residual stream of a multi-layer transformer would carry out. Unlike pass_1
(which only added a temperature and so left targeting identical to the
baseline), this improves **both** histogram sharpness **and** target hit-rate.
`main.py` also evaluates the **n_iter=0 ablation** (refinement removed = plain
matched filter) so the refinement step's contribution to targeting is measured
causally, not assumed.

## Why this visualisation

The **histogram panel** shows three side-by-side attention distributions for a
chosen distractor-similarity slice — mechanism, the no-refinement ablation, and
the dot-product baseline — with the correct key bar in red, so you can see at a
glance that refinement both concentrates the mass and moves the peak onto the
right key. The **sweep panel** is the causal argument: the left axis plots
sharpness and the right axis plots target hit-rate against rising
distractor↔target cosine, with the ablation curve overlaid between mechanism
and baseline. Where the no-refine ablation collapses toward the dot-product
line but the full mechanism stays elevated, the gap *is* the refinement step's
effect — the chance line on the hit-rate axis separates "sharp and right" from
"sharp but wrong".
