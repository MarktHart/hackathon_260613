# attention_equality / pass_4

## What I did

This is a **hand_built** equality-lookup head, expressed as a single
`base_model.py`-style attention layer (QKᵀ → causal mask → softmax, no MLP),
and run entirely in torch on CUDA. The key projections are the **one-hot
token-identity** embedding, so the raw score `Q[i]·K[j]` equals a fixed `temp`
exactly when `tokens[i] == tokens[j]` and `0` otherwise — genuine token
matching. The lookup ("route to the *earlier* occurrence, not yourself") is
added by a single, **position-agnostic** no-self-attention bias that subtracts a
constant from the diagonal of *every* query row; it is identical across all
positions and all sequences and never reads the planted-pair positions `p1`/`p2`.
For the query at `p2`, the only earlier key sharing its token is `p1`, so all
mass routes there by matching alone — yielding `match_mass ≈ 1.0` and
`equality_robustness ≈ 1.0` across the whole `L ∈ {8,16,32,64}` sweep, far above
the shrinking uniform baseline. Unlike the failed pass_3, **no oracle bias keyed
on `p1`/`p2` is used**, so the mechanism actually computes equality. Faithfulness
is shown with two causal ablations: dropping the self-suppression splits mass to
~0.5 (p1 vs self), and zeroing the QK identity collapses routing to ~uniform —
each part is load-bearing.

## Why this visualisation

Three panels, each tied to a claim. **(1) Match mass vs uniform** puts the
goal's exact metric `attn[p2,p1]` on the y-axis against the analytic uniform
baseline across `L`; because the baseline shrinks with `L` while the head stays
at 1.0, the *widening* gap is the real-mechanism signal the README asks for.
**(2) Causal ablations** overlays the full circuit against the two knock-outs on
the same axes — this is the faithfulness evidence, showing the result depends on
the equality match *and* the self-suppression, not on reading the answer.
**(3) Real attention heatmap** displays an actual GPU-computed `L×L` attention
matrix (not a reconstruction) with `p1`/`p2` marked, so a human can directly see
the single bright cell at `(p2, p1)` and confirm the routing is honest.
