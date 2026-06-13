# What I did

This is a **hand_built** attempt: no training, every weight set by hand. The
mechanism is a single cross-attention head expressed as a small delta from
`base_model.py`'s attention — the logits are the sum of two interpretable
terms, `score[i,j] = s_tok · 1[A_i == B_j] − s_pos · |i − j|`. The first term
is the dot product of one-hot token embeddings (a *token-identity gate* that
fires only on equal symbols, the prerequisite for any LCS match); the second
is an ALiBi/T5-style *relative-position bias* encoding the monotone, near-
diagonal prior of an LCS alignment. Together they push almost all of a query's
mass onto the matching key nearest the diagonal — its LCS partner. The four
heads are a built-in **ablation ladder** (token+pos / token-only / pos-only /
uniform), so the sweep *is* the causal test: removing either term collapses
the lift, and removing both reproduces the uniform baseline exactly. All
compute runs in torch on CUDA, and `operating_range.json` sweeps the vocabulary
over 2→64 (≈1.5 orders of magnitude) to show the circuit sharpens as matches
get sparser and degrades when they get dense.

# Why this visualisation

Three coupled views answer the goal's question directly. The **ablation-ladder
bar chart** puts attention-mass-on-LCS-keys on the y-axis with the uniform
baseline drawn as a dashed line, so the full head's lift and the controlled
drop from each ablation are one glance — this is the strawman-vs-working
contrast. The **heatmap** overlays red boxes on the true LCS partner cells, so
a human can verify the bright attention weights actually land on the
DP-computed alignment rather than just being concentrated *somewhere*. The
**operating-range curve** (lift vs. vocab size, log x) shows where the
mechanism holds and where it breaks, which the single canonical number cannot.
