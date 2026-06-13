# attention_dot_product · pass_5

## What I did

This is a **hand-built** attempt (no training): `model_fn` is the exact
scaled-dot-product circuit `softmax(Q Kᵀ / √d_head) · V`, written out by hand as
torch tensors and executed on CUDA. Relative to `base_model.py` the delta is
*minimal* — a single self-attention head's core computation with **no MLP, no
residual, no embedding**, because the task hands `Q`/`K`/`V` directly and asks
only whether the attention output is reproduced. That alone matches the
reference `gt_out` to machine precision (canonical cos_sim ≈ 1.0, mse ≈ 1e-15),
clearing the uniform-attention baseline. The new contribution over a plain
implementation is a **causal ablation study**: I re-run the identical
`task.evaluate` sweep through one parameterised GPU kernel with each component
knocked out in turn — `no_dot` (zero the QKᵀ logits → uniform mixing),
`no_scale` (drop 1/√d), and `no_softmax` (linear normalised weights). Because
the task is synthetic and the mechanism is exact, faithfulness is established
*by construction*; the ablations make that causal claim **measurable** by
showing that removing any single component collapses fidelity toward the
baseline (`no_dot` lands essentially at the uniform strawman; `no_softmax` goes
strongly negative in cosine), which is the on-model analogue of activation
patching for this synthetic setting.

## Why this visualisation

The goal's question is causal — *does the model actually use the dot-product
mechanism?* — so the hero chart is a single horizontal **bar of
`attention_fidelity` per ablation variant**, the exact headline metric the
benchmark optimises. One green bar (full circuit ≈ 1.0) against three red bars
(each component removed) lets a human read "every piece is load bearing" in one
glance, with the benchmark baseline as the implicit zero. The second panel
plots **cos_sim vs sequence length on a log axis** for all variants together:
it answers the goal's robustness sub-question (does fidelity survive growing
softmax competition, 8→128) and simultaneously shows the ablations *staying*
broken across the whole range, not just at one length. The attention heatmap is
the smallest artefact that, if the QKᵀ term were fake, would visibly flip from a
structured pattern to a flat row — a direct check that the weights are the ones
the math claims.
