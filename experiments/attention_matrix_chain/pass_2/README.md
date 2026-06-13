# attention_matrix_chain — pass_2

## What I did

**Hand-built (hardcoded-weights) circuit.** Instead of computing `A2 @ A1` as a
bare matrix product (the prior `first_pass`, which the jury flagged as the
trivial closed form with no mechanism), I express the composition as the
canonical *virtual attention head* construction: a two-layer attention stack
from `base_model.py` with every weight hand-set. The delta from `base_model.py`
is — token/position embedding `E = I` (each position carries a one-hot
"where am I" feature), value projection `W_V = I`, output projection `W_O = I`,
and the residual skip replaced by an overwrite. The given patterns `A1`, `A2`
become the two layers' attention maps, so the residual stream evolves
`X0 = I → X1 = A1·X0 = A1 → X2 = A2·X1 = A2@A1 = A_chain`: stacking two
attention layers literally *writes the composed pattern into the residual
stream*. There is no MLP — attention alone suffices, and I say so. This drives
`composition_robustness ≈ 1.0` (fidelity holds in the peaked regime where the
single-hop shortcut collapses).

**Faithfulness via causal ablation.** I re-run the *same* evaluator with each
layer knocked out: ablating layer 2 makes the circuit emit `A1`; ablating
layer 1 feeds identity straight into layer 2 and emits `A2` (= the single-hop
baseline). Both ablations collapse fidelity at small alpha, showing both hops
are causally necessary — the composition is the observed mechanism, not a
coincidence. All compute (identity embedding, batched attention readouts, OV
projections) runs in torch on CUDA.

## Why this visualisation

The Demo tab makes the mechanism *and* its faithfulness check legible:

- **Heatmaps (per alpha, head 0):** `A1`, `A2`, the circuit's reconstructed
  `A_chain`, the ground-truth `A_chain`, and `|pred − true|`. The eye-test is
  immediate — panel 3 must equal panel 4 (error panel ~0), and at small alpha
  the composed pattern looks nothing like `A2` alone, which is exactly why
  composition matters. Axes are query-position (rows) × key-position (columns),
  the natural layout for an attention matrix.
- **Ablation line chart:** chain fidelity vs Dirichlet alpha for the full
  circuit, layer-2-ablated (→`A1`), layer-1-ablated (→`A2`), and the single-hop
  baseline. The x-axis is the peakedness sweep (peaked/hard regime on the
  left); y-axis is fidelity in `[0,1]`. Only the full two-layer circuit stays
  near 1.0 across two orders of magnitude of alpha, while every ablation
  collapses — the single most informative artefact, since flipping any one
  layer off changes the claim.

The Benchmark tab drops in the shared `benchmark_panel` for cross-attempt
history and the leaderboard.
