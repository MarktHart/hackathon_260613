## What I did

This is a **hand_built** (no-learning) interpretability mechanism for naming a
head's attention mode. Each `(L, L)` matrix is reduced to a few human-named
scalar features — the row-averaged mass on the fixed anchor key
`mean_i A[i,0]` (positional), the diagonal `mean_i A[i,i]`, the `+1` band
`mean_i A[i,i+1]` (induction) and the `-1` band `mean_i A[i,i-1]`
(previous-token), plus row peakiness for context. The classifier is a
**scale-invariant band rule**: a head is the mode whose band carries the most
mass *provided* that band beats a hand-set threshold `TAU = 0.18`; if no band
beats `TAU`, the head is `uniform`. Because the decision is an *argmax across
bands* rather than an absolute distance, it ignores how tall the spike is, so
it survives uniform noise that merely shrinks every spike toward the `1/L`
floor — `mode_robustness` is **1.0** (perfect retention at noise 0.5) with
`accuracy_canonical = 1.0`. All compute runs in torch on `cuda`; the mechanism
is a hand-set readout of the attention layer's output (a 5-way softmax over the
band features), with no MLP and no training — the "`base_model.py` minus the
learned head" delta. As a **baseline**, a strawman that matches absolute L2
distance to the clean prototypes is computed under identical conditions; it
ties the full method on clean data but collapses to **0.56** accuracy at noise
0.5 because the shrinking spikes drift into the uniform prototype's basin.

## Why this visualisation

The Demo tab shows the mechanism end to end for any selected head: the raw
attention **heatmap**, and a **band-mass bar chart with the `TAU` line drawn
across it** — the winning band is highlighted, so you can literally read off the
decision rule and watch the bars sink toward `TAU` as you raise the noise. The
two **probability bars** (full vs strawman) sit side by side for the same head.
Because the headline is robustness, the anchor chart is **accuracy vs noise for
the full rule vs the strawman vs random (0.2)**: the full rule stays pinned at
1.0 across the whole `[0.0, 0.5]` sweep while the strawman peels off at high
noise. That one chart is the testable claim — *scale-invariance, not template
magnitude, is what keeps mode-naming correct under corruption* — and it directly
explains why the band rule beats the naive prototype baseline. The Benchmark tab
drops in the shared `benchmark_panel` so `mode_robustness` and the per-noise
slices are comparable across every attempt at this goal.
