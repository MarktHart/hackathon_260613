# What I did

**Attempt type: hand_built (real QK circuit).** This is `base_model.py` reduced
to a single attention layer (no MLP) plus a few hand-set projection matrices. The
residual stream is `[token-embedding | one-hot position]` of width `d_model = 2L`.
Each of `H = 5` heads is dedicated to one offset `k ∈ {1,2,3,4,8}`: its `W_Q`
reads the positional block as identity (`Q[i] = e_i`) and its `W_K` reads it
through a *shift matrix* `S_k` (`K[j] = e_{j+k}`), so the honest dot product
`Q[i]·K[j] = 1 ⇔ j = i-k`. The attention is then *computed*
(`softmax((X·W_Q)(X·W_K)ᵀ·temp)`) on the GPU — nothing writes mass onto the
target key directly, so the shift-by-k band emerges from the circuit rather than
being hand-painted (the failure mode of pass_2). Because `W_Q`/`W_K` zero out the
token block, the head provably ignores token identity while tokens are still
genuinely present in the stream. I also run a **causal ablation**: replacing
`S_k` with the identity (offset → 0) collapses the target mass to chance, and a
uniform `random_model_fn` strawman is measured under identical conditions. Result:
best-head mass ≈ 1.0 and argmax-accuracy = 1.0 for every `k`, so
`shift_robustness ≈ 1.0`.

# Why this visualisation

The Demo pairs the two artefacts that, if they moved, would break the claim.
The **heatmap** of the dedicated head for a chosen `k` shows attention directly:
a single bright band sitting exactly on the `i-k` sub-diagonal (overlaid in
green) is the shift-by-k operation made visible — query rows put their mass on
the correct earlier key. The **grouped bar chart** answers the goal's real
question — does *some* head implement shift-by-k, and is that the mechanism? —
by putting the real circuit's best-head mass next to two controls per offset:
the ablation (shift matrix removed) and the uniform baseline. The circuit
staying near 1.0 while the ablation drops to the dashed chance line is the
faithfulness evidence; the y-axis is the exact quantity the benchmark scores
(mass on key `i-k`), and the x-axis is the full offset sweep so robustness is
legible at a glance.
