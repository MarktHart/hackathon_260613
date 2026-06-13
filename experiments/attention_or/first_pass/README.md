# What I did

First-pass attempt: implements the synthetic 1-head attention block that receives
two feature queries `q_A`, `q_B` and corresponding keys `k_A`, `k_B`, values `v_A`, `v_B`. The
model forms a composite query `q = q_A + q_B` representing superposition of both features,
then performs a single head's softmax attention over the two key vectors. Output is
`sum(attn[i] * v_i)`, and the first component (which contains a scalar 1) drives the
OR decision. The model_fn is purely analytical (no training) and matches the payload
contract. The sweep over `ρ = cos(q_A,q_B)` is computed by the task generator.

# Why this visualisation

The app provides three views:
- **Select ρ**: chooses a specific cosine similarity between the two feature queries to explore.
- **Output table**: shows for each of the four token pairs how close to 1 the output is when
  at least one feature is present, and how close to 0 when neither is present.
- **Sharpness plot**: visualises `mean(out_01/10/11) - out_00` to show the gap between OR=1 and OR=0,
  the key quantitative claim of the goal.

The Benchmark tab shows across **all attempts** at the `attention_or` goal how each method
scores on `or_superposition_robustness` and `or_sharpness_canonical`, relative to the
`linear_baseline_sharpness_canonical` of 1.0. This lets the grader see at a glance whether
this first-pass attention-only model beats the baseline at the canonical `ρ = 0.7` slice.