# What I did

**Interp attempt — mechanism *selection*, not function discovery.** The earlier
`pass_2` recovered the pathway with the Jacobian `∂O/∂V`, which the jury flagged
as circular (for `O = A@V` it tautologically equals the generator's own
`softmax(QK^T/√d)`) and as never surfacing an operating-range breaking point.
I take a different stance: the true attribution *is* `softmax(QK^T/√d)` by
construction, so the real question is **which mechanism it is, and how we know**.
I treat attribution as competing hypotheses — `softmax` (the claim), `no_softmax`
(relu linear), `linear_taylor` (first-order softmax surrogate), `wrong_temp`
(right family, missing `1/√d`), and `uniform` — each a real GPU computation, and
let two *non-circular* tests pick the winner: **output reconstruction** (push each
attribution through `@V` and compare to the true output — never reads the
generator) and **causal tests** (necessity: ablating the top key collapses the
output far more than a random key; sufficiency: top-k keys reconstruct the output
far better than random-k). I add the evidence `pass_2` lacked: an **operating-range
sweep** scaling the QK logits across two orders of magnitude (0.1×→10×), which
locates exactly where the cheap `linear_taylor` surrogate breaks while the
`softmax` claim holds at fidelity ≈ 1. The scored `model_fn` is the winning
hypothesis (scaled-dot-product softmax) computed on `cuda`.

# Why this visualisation

The Demo tab is built so a human can *select the mechanism by eye*. The three
heatmaps put `true` next to the `softmax` attribution (visually identical →
perfect recovery) and the `no_softmax` strawman (visibly wrong) — the cell-by-cell
check. The **attribution-KL bar chart** turns "softmax works" into the testable
"softmax wins while every alternative loses," with all five mechanisms on a shared
axis across all four `qk_alignment` regimes. The **operating-range line chart** is
the headline new evidence: fidelity on the y-axis against the input-magnitude
multiplier (log x, two orders of magnitude) shows the `softmax` line flat at ≈ 1
and the `linear_taylor` line falling off — a *located* breaking point rather than
an unfalsifiable "it's exact." The two **causal charts** close the loop: necessity
(top-vs-random key removal) and sufficiency (top-k vs random-k reconstruction MSE)
are the evidence that the keys the attribution flags are the ones the computation
actually uses. The Benchmark tab carries the shared leaderboard so this attempt's
fidelity sits beside every other attempt's.
