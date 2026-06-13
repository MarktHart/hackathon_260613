# attention_and / superposed_query

## What I did

Built a 7-token toy setup in a 2D concept space `(e_A, e_B)`. Each token has
hand-chosen `a_match` and `b_match` inner products with the concept directions
(two tokens only-A `(3, -1)`, two only-B `(-1, 3)`, one `both (3, 3)`, two
`neither (-1, -1)`). The joint query is the superposition `q = q_A + q_B`, so
each token's pre-softmax score is exactly `scale * (a_match + b_match)`. I
sweep `scale ∈ [0.25, 3]` and dump, per token and scale, both the softmax
attention weight and a "linear baseline" — the same scores shifted to be
non-negative and renormalised to a distribution. The linear baseline is the
counterfactual answer to *what would attention look like without the `exp`?*

## Why this visualisation

The goal's claim is: under softmax, a superposed query becomes a product of
per-concept match scores, so attention picks the conjunction. That claim has a
single observable consequence — the `both` token's mass should *visibly*
dominate after softmax but stay roughly tied with the only-A and only-B
tokens under the linear baseline. The grouped bar chart puts those two
distributions side by side on the same axes, so the AND-sharpening *is* the
biggest pixel difference in the plot. The `scale` slider exposes the second
half of the claim — that the sharpening is monotone in the score scale, which
is what you'd expect from a product of exponentials — so the grader can watch
the AND-spike rise from "slightly preferred" to "essentially all the mass".
The dashed uniform line is a calibration: any bar above it is winning,
anything below losing, and you can read off the absolute degree of preference
at a glance.
