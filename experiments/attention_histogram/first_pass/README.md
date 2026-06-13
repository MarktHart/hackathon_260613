# attention_histogram / first_pass

## What I did

This is a **hand_built** attempt — no training, the only knob is a hand-set
inverse-temperature β (default 12). The mechanism is the smallest possible delta
from `base_model.py`: that file scores keys against the query with a matched
filter (`q @ kᵀ`) and applies the fixed `1/sqrt(head_dim)` temperature inside
`scaled_dot_product_attention`. I keep the identical matched-filter score —
cosine-normalising q and k so scores live in `[-1,1]` — because the matched
filter is the optimal way to pick the target direction out of distractors when
the query is a noisy copy of the target. I then replace the fixed temperature
with a single tunable β that *sharpens* the softmax. Larger β lowers the
histogram entropy without changing the argmax, so the head stays correctly
aimed (same matched-filter targeting as plain dot-product) while producing a
far lower-entropy, single-peaked attention distribution. The real compute runs
in torch on CUDA; the linear (no-mechanism) dot-product baseline is computed by
the evaluator under identical conditions for comparison.

## Why this visualisation

The **histogram panel** is the literal object the goal asks about: side-by-side
attention bars for the mechanism vs. the dot-product baseline at a chosen
distractor-similarity slice, with the correct key bar coloured red — you can see
at a glance whether the mass is single-peaked and on the right position
("sharp and right") versus smeared ("high entropy") or peaked elsewhere
("sharp but wrong"). The **sweep panel** puts the right thing on each y-axis:
left plots histogram sharpness (`1 − H/log n`, the metric) against rising
distractor↔target cosine so degradation under interference is visible as a
falling curve, with the baseline overlaid to show the lift; right plots target
hit rate vs. the same axis with the `1/16` chance line, separating sharpening
from correct aiming. Together they let a human verify both claims the headline
`histogram_robustness` and `mean_target_hit_rate` metrics encode.
