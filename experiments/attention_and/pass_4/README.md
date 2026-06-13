# What I did

**Type: hand_built.** I express the attention score as a *multiplicative*
conjunction of two feature read-offs: `logit = β · relu(residual·q_A) ·
relu(residual·q_B)`. This is a one-line delta from `base_model.py`'s dot-product
attention score — the bilinear product is the smallest nonlinearity that
implements a true logical AND, since either factor being ~0 forces the logit to
0. The ReLU clips negative noise so "neither" positions stay near zero instead
of producing a spurious positive product (the failure mode of a plain product).
The task's own `linear_baseline` (`residual·q_A + residual·q_B`) is the strawman:
being additive, it lights up single-feature positions, so the product head's
`lift_over_baseline_canonical` is the causal evidence that the *multiplication*,
not the projections, is what gates the AND. All compute runs in torch on CUDA.

Faithfulness note: this is a synthetic hand-built circuit, not a probed trained
model. The built-in causal check is the baseline swap — replacing the product
with a sum (no AND gate) collapses sharpness, which the Demo tab shows directly.
Robustness to superposition holds because an `A AND B` position accumulates
`2·q_A + 2·q_B`, giving a strictly larger projection magnitude than either
single feature even when `q_A = q_B` at cos = 1.0.

# Why this visualisation

The Demo bar chart puts **mean attention mass per condition** on the y-axis for
the four presence cases `(neither, A-only, B-only, A∧B)`, the exact quantity the
AND claim is about. A correct head shows one tall red bar at `A AND B` and three
bars pinned near the dashed uniform line; the linear-baseline radio flips to the
strawman so the grader watches the single-feature bars rise. The cosine slider
sweeps superposition so degradation (if any) is visible as the off-condition
bars creeping up. The Benchmark tab tracks `superposition_robustness` and
`lift_over_baseline_canonical` across attempts.
