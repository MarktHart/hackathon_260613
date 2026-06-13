# attention_block_2d — pass_5

## What I did
**Hand-built (interp) — a real attention block plus a geometric reader.** Unlike
pass_4 (a matrix reader with only a *described* ablation), this attempt builds an
actual model so the faithfulness claim is *run*, not argued. The **producer**,
`Block2DAttention`, is a single-head 2D attention block — a small delta from
`experiments/base_model.py`'s `Attention`: the content score `q·kᵀ` is switched
off and the spatial structure lives in a hand-set **additive bias** — the
Swin-style relative-position-bias table for `local`/`dilated`, a global-token
bias row+column for `global`, and a causal mask for `causal_2d`;
`softmax(bias + tiny noise)` gives the row-stochastic matrix. The MLP is dropped
(it does nothing for a pure attention-shape task). The **reader**, `classify`,
is a hand-built geometric method that recovers `(family, params)` from a matrix
alone — key−query displacement offsets → window size & dilation, a full
row+column over otherwise-sparse rows → global token, a triangular mask → 2D
causal. It never peeks at the producer's bias or the task's private generators,
and scores all 16 canonical examples correctly (lift `+0.75` over the majority
baseline). On top of the headline metric, `main.py` runs two checks pass_4 only
promised: a **causal ablation** — zeroing the producer's bias table collapses
attention to uniform and drops reader accuracy `4/4 → 0/4`, proving the pattern
*causally* lives in that table — and an **operating-range sweep** driving the
producer across grids from 4×4 (N=16) to 32×32 (N=1024), a 64× span, where the
reader stays at ~1.0 while the majority baseline sits at ~0.2.

## Why this visualisation
Four views, each falsifiable by eye. **Demo** shows the matrix beside its
*displacement footprint* — the exact `(dr, dc)` quantity the reader consumes —
so a tight blob reads `local`, a spaced blob `dilated`, a cross `global`, a
half-plane `causal_2d`; the per-family bar puts the claim against the dashed
majority baseline. **Faithfulness** is the core addition: intact-vs-ablated
heatmaps for any family, the reader's verdict under each, and an accuracy bar
collapsing from intact to ablated — this is the causal evidence that the bias
table *is* the mechanism, the thing pass_4 lacked. **Operating range** plots
reader accuracy vs grid size on a log-N axis against the baseline, so the
grader sees the method hold across two orders of magnitude rather than only the
canonical 16. **Benchmark** tracks `pattern_acc_canonical` and per-family
accuracies across all attempts.
