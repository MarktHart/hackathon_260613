# What I did

**Hand-built / interp** attempt (no training) — a 2-layer, attention-only model
that is `base_model.py` with one real delta: the summation head drops softmax.
**Layer 1** is a textbook softmax *copy* head: every target position attends
sharply to the coefficient token at position 2 (weight ≈ 1) and copies (α, β)
into its own residual — the broadcast is **computed by attention**, not placed
by hand (the fix for the prior attempt's fudge). **Layer 2** is the identical
head with **softmax removed (linear attention)**: its query reads the
now-local (α, β), one-hot position-identity keys give `score(t,0)=α`,
`score(t,1)=β`, and the value carries only the scalar feature, so the
un-normalised weighted sum `Σ_j score·v_j = α·x₁ + β·x₂` is exact for every
coefficient — including negative and magnitude-32 values. Coefficients touch
only the Q/K path of the summation head; the value never sees them. I prove
the mechanism causally by running the **same forward pass** in three modes:
*linear* (R²≈1), *softmax_sum* (the strawman — fails even at α=β=1 because
convex weights sum to 1), and *broadcast-ablated* (zero layer-1 output → the
query loses (α,β) → output collapses), confirming both heads are necessary.

# Why this visualisation

The Demo figure is built so the claim survives without the README. The **bar
chart** is the single decisive comparison the goal asks for — canonical R² of
the mechanism vs. the softmax strawman vs. the broadcast-ablated circuit vs.
the mean baseline — so "linear attention works *where softmax and the ablation
don't*" is one glance, not a paragraph. The **operating-range line** puts R² on
the y-axis against `|α|=|β|` on a log x-axis from 0.25 to 32 (>2 orders of
magnitude): linear stays pinned at 1, softmax decays — exactly the axis the
goal's robustness metric hinges on. The two **R² heatmaps** over the full 24
(α, β) grid (α rows, β cols, shared −1…1 scale) show the mechanism is green
everywhere while the strawman goes red wherever the convex-combination
constraint bites. The Benchmark tab tracks these metrics across attempts so
iteration reads as a curve.
