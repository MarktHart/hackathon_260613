## What I did

This is an **interp / hand_built** attempt that fixes the previous one's tautology
(first_pass directly emitted the ground-truth sub-diagonal matrix). Here
predecessor attention is *computed* by a genuine single-head QK circuit — a
minimal delta from `base_model.py`: keep the token embedding and one
self-attention head, **drop the MLP**, concatenate sinusoidal **position
features** onto the residual stream, and hand-set `W_Q` to a block-diagonal
**rotation-by-δ** on the positional sub-space with `W_K` = identity there (both
zero the token sub-space). The score is then
`score(t,s) = (R_δ p_t)·p_s = p_{t+δ}·p_s = Σ_k cos(ω_k(t+δ−s))`, a kernel that
peaks at `s = t+δ`; with `δ=-1` the softmax selects the immediate predecessor
through the dot-product, not by writing the answer. The benchmark head uses
`δ=-1` and runs entirely in torch on CUDA. `main.py` also saves three controls:
a **random-attention strawman**, **ablations** (zeroing positions or the
rotation collapses alignment to baseline, while zeroing/shuffling tokens leaves
it untouched — causal evidence the head is positional), and an **operating-range
sweep** over `T = 8 … 512`.

## Why this visualisation

The Demo tab leads with the **offset × data-shift matrix**: rows are the head's
rotation δ, columns are the data's true shift, cells are alignment. A bright
diagonal is the falsifiable signature that the *same QK circuit* tracks whatever
shift it is rotated to — exactly what "the mechanism computes the target"
means, and impossible to fake by emitting one fixed matrix. The **canonical
sweep** bar chart puts the predecessor head (δ=-1) against the uniform baseline
and the random strawman so the ~1.0-vs-~0 contrast is legible at the measured
metric (`mean_max_attn_to_target`). The **ablation** chart is the faithfulness
check: blue bars (circuit intact, incl. shuffled tokens) stay high, orange bars
(positions or rotation removed) drop to the uniform line — showing *which*
sub-circuit causes the behaviour. The **operating-range** line is deliberately honest: the head is perfect to
`T≈64` then degrades smoothly (≈0.52 at 128, ≈0.13 at 512) because the fixed
budget of 32 positional frequencies runs out of angular resolution as the
sequence grows — a graceful, interpretable failure (more frequencies would push
the limit out), not a silent collapse. The example heatmap shows the literal
sub-diagonal band the circuit produces at the canonical `T=32`.
