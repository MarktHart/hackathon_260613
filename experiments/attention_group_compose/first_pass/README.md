# attention_group_compose — first_pass (snap-to-group)

## What I did
This is a **hand_built / interp** attempt (no training). The claim is that
noisy attention matrices representing elements of the cyclic group C₆ compose
according to the **group law**, not according to softmax-relaxed matrix
multiplication. The `model_fn` is a hand-set circuit: it scores each noisy
input against the `n` rotation templates `P_k` (an einsum overlap on CUDA),
takes the argmax to recover each element's rotation index `k_a, k_b`, composes
them exactly in the group as `k_c = (k_a + k_b) mod n`, and returns the clean
permutation `P_{k_c}`. There are no learned weights — the "circuit" is just the
rotation templates plus the modular-addition composition rule, i.e. the group
structure (closure + the C₆ Cayley table) made explicit. Relative to
`base_model.py` this uses zero blocks: a single attention-style template match
(the same `softmax`/argmax read-out an induction head performs) is sufficient,
so the MLP and residual stack are dropped. The naive `A@B` baseline is computed
by `task.py` under identical conditions, so the **lift over baseline** at
canonical σ=20 is the headline test. A natural faithfulness check (future work,
since this attempt is synthetic) would be to ablate the group-composition step —
replace `(k_a+k_b) mod n` with raw template scores — and confirm fidelity
collapses to the matmul baseline.

## Why this visualisation
The Demo tab leads with **fidelity vs noise**, plotting the snap-to-group method
against the naive-matmul baseline on the same axes with the canonical σ=20 line
marked. That is exactly the goal's question — does structure-aware composition
beat the relaxation? — reduced to two curves: when the blue curve sits above the
red one, the group law is doing real work, and the vertical gap at σ=20 *is* the
`lift_over_baseline_canonical` metric. The y-axis is fidelity (1 − normalised
Frobenius error) so it reads directly against the benchmark, and the x-axis
spans the full clean→chance sweep so the operating range is visible. The heatmap
row (A, B, predicted C, true C) lets a human eyeball one concrete composition
and confirm the prediction *is* a clean permutation matching the true group
product — not a fuzzy average like naive matmul produces. The Benchmark tab
carries the shared cross-attempt leaderboard and metric history.
