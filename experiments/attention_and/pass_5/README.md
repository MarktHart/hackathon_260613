# attention_and · pass_5 — magnitude AND head

## What I did

This is a **hand_built** attempt (no training): a single attention head whose
weights are set by hand. The head reads the two QK probes `a = ⟨r, q_A⟩` and
`b = ⟨r, q_B⟩` off the residual stream, forms the cosine-corrected magnitude
`S = (a + b) / (2(1 + cos))` — an estimate of *how many of the two features are
present* — and passes it through one gating nonlinearity
`logit = GAIN · σ(STEEP · (S − 1.5))`. The gate lights up only when `S ≈ 2`,
i.e. **both** features present, so after the softmax in `task.evaluate` the head
attends sharply to exactly the AND positions. Relative to `base_model.py` the
delta is tiny: the QKV projection reads `q_A, q_B`, and the squared-ReLU MLP is
collapsed to a single hand-set gate — attention plus one nonlinear unit, one
block, no training. The key idea (and the fix over pass_4) is to threshold the
**combined magnitude** rather than a per-feature product: when `cos(q_A,q_B)→1`
the two directions merge and A/B become individually unidentifiable (the 2×2
Gram matrix is singular, which is exactly where pass_4's product gate collapsed
to robustness 0.000), but the *count* of features is still readable from the
magnitude, so the AND boundary survives. Measured: sharpness ≈ 0.996 flat across
the whole cosine sweep, **superposition_robustness = 1.00**, lift over the linear
baseline ≈ +0.23. `main.py` also evaluates three ablations (no gate, no
`(1+cos)` correction, per-feature product) and writes their sweeps as artefacts.

Faithfulness note: this is a purely synthetic, hand-set circuit, so there is no
trained model to ablate. Instead the causal argument is made *within the
mechanism* — each ablation knocks out one component and the robustness curve
shows the corresponding failure. A model-level check would train the
single-block transformer on this AND task and ablate the gating unit / the
`(1+cos)` scaling to confirm a learned head recovers the same magnitude circuit.

## Why this visualisation

The headline question is *robustness to superposition*, so the primary chart
puts **AND sharpness on the y-axis against `cos(q_A,q_B)` on the x-axis** and
overlays five lines on one set of axes: our head, the linear baseline (the
goal's no-mechanism reference), and three ablations. The claim "AND survives
superposition" is legible at a glance as the flat top line, while every line
that *dives toward cos = 1* names a specific design choice you cannot remove —
in particular the per-feature product gate (pass_4) crashing to 0 makes the
contrast with this attempt concrete. The second panel zooms into *why* it works:
at a chosen cosine it scatters the gate input `S` coloured by the ground-truth
AND label with the decision threshold drawn in, so a human can directly see the
"both" cluster sitting at `S≈2` cleanly separated from "one/none" — the
mechanism, not just its score.
