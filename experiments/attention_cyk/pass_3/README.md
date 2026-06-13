# attention_cyk — pass_3

## What I did
This is a **hand_built / interp** attempt: `base_model.py` minus the MLP, with
**a single attention head** that finds the CYK split point — no learning, no
data-dependent Python routing. A causal counting head (strict lower-triangular
attention over the sign embedding `(`→+1, `)`→−1) produces the prefix bracket
depth `D(p)`. The split head then scores candidate split `k` for query cell
`(i, j)` with one linear QK score,
`score(k) = −T·(D(k)−D(i))² + β·(D(i)+0.5−D(k))·k`, which is exactly
`⟨q(i), φ(k)⟩` with `φ(k) = [D(k), D(k)², k, k·D(k)]`. The quadratic snaps
attention onto the depth level `D(i)`; the position term resolves *which* point
on that level — and because the linear term is ≥0 at level `D(i)` and ≤0 above
it (with the quadratic adding `T` on top), the two regimes provably never
interfere for any `β`. So one head handles all three filled-cell types: it
picks the **latest** depth-`D(i)` crossing for `S→S S` and `X→S R` cells (always
a correct split), and the **earliest** min-level point `k=i+1` for wrapped
`S→L X` cells that have no crossing. This fixes pass_2's flagged weakness — its
hand-coded `if span_balanced / elif X / else wrap` dispatch is gone, replaced by
one genuine attention score. It reaches **1.0000** canonical split accuracy at
every span length (lift ≈ +0.63 over the uniform 0.37 baseline, robustness =
1.0). All compute runs in torch on `cuda`. As **faithfulness/causal checks** I
ablate the two ingredients: zeroing the position term (β=0, pure depth-matching)
drops accuracy to ~0.84 — it can no longer single out `j-1` among several
balance points; zeroing the depth feature (`D:=0`) drops it to ~0.55 — the score
becomes position-only and loses the `S` cells. The same checks on a *trained*
head would be activation-patching the depth feature and the position channel.

## Why this visualisation
The Demo overlays the two quantities the claim depends on, on shared
split-position axes. The top panel is the string's bracket-depth profile with a
dashed reference line at the cell's start depth `D(i)`; the bottom panel is the
head's actual attention distribution over candidate splits, with CYK-correct
splits coloured green. Cycling the example dropdown through the three cell types
(`S→S S`, `X→S R`, wrapped `S→L X`) lets a human verify the *single* head does
the right thing in each regime: it spikes on the latest `D(i)`-crossing for the
first two and on `k=i+1` for the wrap — the visual *is* the argument that one
score covers every case. The companion bars give the headline plus **two
ablations** (`no-position` and `depth-ablated`), each isolating one named
ingredient, and the per-span line shows the result is flat at 1.0 from span 3 to
8 rather than cherry-picked.
