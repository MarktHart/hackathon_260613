# attention_argmin / pass_3

## What I did

**Attempt type: hand_built (interp).** This is `base_model.py` reduced to a
single attention head with no MLP, whose `W_Q`/`W_K` are set by hand rather than
trained. The mechanism is exact and stated in one line: each 32-dim key is
augmented with a 33rd channel carrying the position's scalar value, and the
effective query is zero on the random key dims and `-beta` on the value channel.
The attention logit is therefore exactly `logit_i = -beta * value_i`, computed as
a real query·key matmul on the GPU. Because softmax concentrates on the
*largest* logit, the negative sign makes it concentrate on the *smallest value* —
the argmin — with `beta` (inverse temperature) controlling how sharply. The
scored head uses `beta = 20`, which beats the uniform strawman (`sharpness ≈ 1`)
by a wide margin at the canonical gap. `main.py` also sweeps `beta` and saves it:
`beta = 0` collapses the head exactly to the uniform baseline, and raising `beta`
monotonically sharpens the argmin — this *is* the causal/ablation evidence that
the `-beta·value` channel is the mechanism (knock it out → behaviour gone).

## Why this visualisation

The Demo tab shows the two things that make the claim checkable. (1) A bar plot
of attention over all 64 positions for a sampled sequence, with the true argmin
in red, the runner-up in orange, and the uniform `1/64` line dashed — you can see
the mass collapse onto the minimum and compare it directly against the
no-mechanism reference. (2) A sharpness-vs-`beta` curve (sharpness =
`attn@min × 64`, exactly the benchmark's headline metric) with the uniform
baseline at 1.0: it starts at 1.0 when `beta = 0` (ablated head) and climbs
toward 64, making the causal role of `beta` legible as a curve rather than a
single number. The `gap` slider lets the grader sweep task difficulty across the
canonical range and watch where the head degrades. The Benchmark tab carries the
shared cross-attempt leaderboard.
