# What I did

**Type: hand_built (analytic, no training).** I built a single **multi-head
self-attention block** — the smallest delta from `base_model.py` — whose weights
are set by hand from the pattern and embedding matrix. There is **one head per
concrete (non-wildcard) pattern offset** `j`. Each head uses a sharp
*relative-position* bias so its softmax collapses to a one-hot that gathers
exactly the neighbour `t = i-(L-1-j)` of query position `i` (a genuine
`softmax(QKᵀ)·V` over an `N×N` score matrix, run on CUDA). The value each head
reads is the token-match score `residual[t]·embed[pattern[j]]`. The **only**
change to a standard transformer block is the head readout: instead of
concat+project I combine heads with an **element-wise min**, which is logical
**AND** in score space — a window is a match-end only if *every* concrete offset
matches. This fixes the previous attempt's flaw, where the README claimed AND
but the code *summed* similarities (an OR-ish combine that leaked false
positives as `L` grew). Result: `length_robustness = 0.999`, canonical
sharpness `0.98`, lift `+0.54` over the linear baseline, with FPR/FNR held low
across `L = 1..6`. The L=1 case degenerates exactly to single-head content
matching (== the baseline), which is the correct mechanistic story: 1-grams
need no composition; multi-token patterns do.

*Faithfulness note:* this is a **synthetic** attempt — there is no trained model
to ablate. The causal claim is structural: knocking out any single head (set
its contribution to `+∞` so it drops out of the `min`) removes one offset's
constraint and should reintroduce false positives at every window that matches
the *other* offsets — the AND-collapses-to-OR signature. That head-ablation is
exactly the check a trained version of this circuit would need.

# Why this visualisation

The Demo tab has two panels aimed at the goal's two questions. **(A)** A single
example overlays per-position attention with the true match-end positions (red
dots) and the uniform `1/N` decision threshold, so a human can see at a glance
that mass lands on *where the pattern finishes matching*, not on every
occurrence of its final token. **(B)** The sweep-summary panel — promised but
missing from the prior attempt, now implemented — plots `match_sharpness`
against pattern length for the matcher and the no-composition linear baseline
(the vertical gap is the lift; the endpoint ratio is the headline
`length_robustness`), alongside FPR/FNR vs `L`. That second axis is the whole
point: a final-token detector collapses as `L` grows while a true compositional
AND stays flat, and the FPR curve shows min/AND not leaking where summing would.
