# attention_and · pass_6 — magnitude AND head

## What I did

This is a **hand_built** attempt (no training). A single attention head reads
two QK probes off the residual stream, `a = ⟨r, q_A⟩` and `b = ⟨r, q_B⟩`, forms
the cosine-corrected magnitude `S = (a + b) / (2(1 + cos))` — an estimate of
*how many of the two features are present* (S ≈ {0,1,2}) — and passes it through
one gating nonlinearity `logit = GAIN·σ(STEEP·(S − 1.5))` that fires only when
`S ≈ 2`, i.e. **both** features present. Relative to `base_model.py` the delta is
tiny: the QKV projection reads `q_A, q_B`, and the squared-ReLU MLP is collapsed
to one hand-set gate — attention plus one nonlinear unit, no training. The key
idea (and the fix over a per-feature product gate) is to threshold the *combined
count* rather than a product: when `cos(q_A,q_B) → 1` the directions merge and
A/B become individually unidentifiable (the 2×2 Gram matrix is singular, exactly
where the product gate collapses to robustness 0), but the count of features
stays readable, so the AND boundary survives. `main.py` runs the head plus two
ablations (drop the gate, per-feature product) so the failure modes are explicit.

Faithfulness note: this is a purely synthetic hand-set circuit, so there is no
trained model to ablate. The causal argument is made *within the mechanism* —
each ablation removes one component and the robustness curve shows the matching
failure. A model-level check would train the single-block transformer on this
AND task and ablate the gating unit / the `(1+cos)` scaling to confirm a learned
head recovers the same magnitude circuit.

## Why this visualisation

The goal asks about *robustness to superposition*, so the primary chart puts
**AND sharpness on the y-axis against `cos(q_A,q_B)` on the x-axis** and overlays
our head, the goal's linear baseline, and two ablations on one set of axes. The
claim "AND survives superposition" reads as the flat top line; every line that
dives toward `cos = 1` names a design choice you cannot remove — in particular
the per-feature product gate crashing makes the contrast concrete. The second
panel zooms into *why* it works: at a chosen cosine it scatters the gate input
`S` coloured by the ground-truth AND label with the threshold drawn in, so the
"both" cluster at `S ≈ 2` is visibly separated from "one/none".
