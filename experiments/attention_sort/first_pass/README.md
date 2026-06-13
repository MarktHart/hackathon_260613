# attention_sort / first_pass

## What I did

**Hand-built** (`interp` / hardcoded-weights) sorting head — a single
attention block, a minimal delta from `base_model.py` (one attention layer, no
MLP, no learned weights). It sorts in two value-driven steps. **(1) Rank by
counting:** a uniform-weight comparison head reads the pairwise feature
`v_j − v_k` and sums `sigmoid(τ·(v_j − v_k))` over keys, so each position's
attention-mass count approximates its sorted rank (number of values below it).
**(2) Route by rank:** output slot `i` scores key `j` with
`logit[i,j] = −β·(rank_j − i)²`, so its argmax lands on the position whose rank
is `i` — exactly `argsort(values)[i]`. Both steps depend only on value
*comparisons*, never on absolute position, so the **same fixed `(τ, β)`
generalise across every length** — giving `sort_robustness ≈ 1` where a
left-to-right positional shortcut would collapse. The benchmark uses a sharp
`τ=1e4`; `main.py` also sweeps `τ` to show the mechanism is the sharp-comparison
limit of a soft counting head.

The strawman is built into the goal's metrics: uniform/random attention scores
`1/L` (plotted as the dashed reference) and decays, whereas this head holds near
`1.0` from `L=4` to `L=32`. A causal/faithfulness note: this is a *synthetic
hand-set circuit*, not a trained model, so there is no learned weight to ablate;
the natural faithfulness check is the **τ-ablation** already shown — softening
the comparison head (small `τ`) destroys the rank signal and accuracy falls to
the uniform floor, confirming the counting step is load-bearing.

## Why this visualisation

The **heatmap** puts output slots on `y` and input positions on `x` with cyan
circles at the `argsort` target; the claim "the head sorts" is true iff the
bright mass sits on the circles, checkable at a glance and at any length. The
**accuracy-vs-length** line (log-2 `x`) against the `1/L` uniform baseline is the
exact picture `sort_robustness` summarises — a flat line near 1 means genuine
sorting, a decaying one means a positional shortcut. The **accuracy-vs-τ** curve
is the ablation: it shows accuracy rising from the uniform floor to ~1 as the
comparison head sharpens, isolating the counting step as the mechanism that
matters.
