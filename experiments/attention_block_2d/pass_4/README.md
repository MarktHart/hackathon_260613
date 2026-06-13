# attention_block_2d — pass_4

## What I did
**Hand-built (interp) geometric classifier.** This is *not* a learned model and
— unlike the earlier pass_3 — it never reconstructs ground-truth matrices from
the task's private generators. Instead it reads the spatial pattern straight off
the attention matrix geometry, on the GPU. Steps: (1) binarise with an absolute
threshold `0.4/N`, which separates the ~`1/k` allowed mass from the sub-`ε`
noise floor; (2) **global** if some index `p` has both a fully-attended row
(token attends everyone) and a fully-attended column (everyone attends it);
(3) **causal_2d** if the binary mask equals the lower-triangular raster mask;
(4) otherwise **local/dilated** by collecting the distinct positive
`key−query` displacement offsets — the smallest offset is the *dilation* and the
number of offsets is the *window radius*, so `dilation==1 → local`,
`dilation>1 → dilated`. Confidence is the fraction of attention mass landing on
the reconstructed ideal window/global/causal mask. The method gets all 16
canonical examples correct (lift `+0.75` over the majority-`local` baseline of
`0.25`), and because every rule is a thresholded geometric statement it doubles
as a faithfulness check: zeroing the displacement read-out collapses
local/dilated, and removing the full-row+column test collapses global —
each rule is the minimal thing whose removal breaks one family.

## Why this visualisation
The Demo tab's key panel is the **displacement footprint**: attention mass
binned by `(dr, dc) = key−query` offset on a 15×15 grid centred at the query.
This is the exact quantity the classifier consumes, so the human sees the
evidence, not a proxy — a tight 3×3 blob reads as `local`, the same blob spaced
two apart reads as `dilated (dilation=2)`, a cross reads as `global`, and a
filled half-plane reads as `causal_2d`. Beside it the raw matrix heatmap lets
you confirm the footprint came from the structure you expect. The per-family
accuracy bar puts the four families on one axis against the dashed majority
baseline, so the claim "geometry recovers every family, the strawman only gets
local" is legible in a single glance. The Benchmark tab tracks
`pattern_acc_canonical` and the per-family accuracies across all attempts.
