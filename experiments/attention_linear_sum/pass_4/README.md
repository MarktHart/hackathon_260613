# What I did

**Hand-built (interp)** attempt — no training. This is `experiments/base_model.py`'s
`Attention` with one deliberate delta: **softmax is removed (linear attention)** and
all of Q/K/V/O are hand-set. The residual stream places `x₁` at pos 0, `x₂` at pos 1,
and the coefficient token `(α, β)` at pos 2, which is broadcast into every target
position's query slot (the goal's "coefficients supplied only in the Q/K projections").
The query reads `(α, β)`; position-identity keys make `score(t,0)=α`, `score(t,1)=β`,
zero elsewhere; the value carries the scalar feature; so the un-normalised weighted
sum `Σ_j score·v_j = α·x₁ + β·x₂` is **exact** for every coefficient, including
negative and magnitude-2 values. I run the **softmax version of the identical head**
as a strawman — softmax weights are non-negative and sum to 1, so they cannot express
`|α|+|β|≠1` or negative coefficients. This yields R²≈1.0 canonical and robustness≈1.0,
while the softmax head fails on most of the 24-pair sweep.

# Why this visualisation

The Demo tab is two side-by-side **R² heatmaps over the full 24 (α, β) sweep** — the
linear-attention head vs. the softmax strawman — on a shared −1…1 colour scale, with
α on the rows and β on the columns. That grid is exactly the axis the goal's
robustness metric hinges on, so a human can read correctness *and* operating range in
one glance: the linear head is green (≈1) across every coefficient, while the softmax
head goes red wherever the convex-combination constraint bites. The third panel is a
pred-vs-target scatter on the canonical α=β=1 condition with the identity line, the
smallest artefact that, if the mechanism were wrong, would visibly bend off the
diagonal. The Benchmark tab tracks these metrics across attempts so iteration shows up
as a curve.