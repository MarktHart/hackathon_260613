# What I did

This is a **hand-built (interp)** attempt. It reframes attention saturation as
**softmax-Jacobian collapse**: the per-query softmax Jacobian is `J = diag(p) − ppᵀ`
with trace `1 − Σp²`, so the attention concentration `C = Σp²` is exactly *one
minus the gradient flow*. I use `saturation_score = mean Σp²` — a quantity that
is mechanistically the vanishing-gradient signal the goal names, not just a
restatement of the reference max-weight. All forward and **autograd** compute runs
in torch on CUDA: I additionally measure the real gradient `∂‖attn·v‖²/∂logits`
and show it collapses by orders of magnitude exactly where `C → 1`, which is the
causal/faithfulness evidence that the detector and the gradient death are the
same phenomenon. The key novel piece is a **failing strawman — ablate the `exp`**:
a relu-linear (no-exp) head, `wᵢ = relu(sᵢ)/Σrelu(sⱼ)`, is algebraically
*scale-invariant* (the `logit_scale` factor cancels), so it cannot saturate, its
concentration is flat across the full 0.1→100 sweep, and its AUROC collapses to
chance — proving `exp` is the mechanism. The headline softmax head still matches
the oracle (AUROC ≈ 1.0, entropy correlation ≈ 1.0).

Faithfulness note: the task is purely synthetic with no trained model to patch,
so the causal check here is the architectural ablation (remove `exp`) plus the
direct autograd gradient measurement. On a real model the analogous check would
be patching the attention logits' temperature / clamping logit magnitude and
watching the downstream behaviour and gradient signal both break.

# Why this visualisation

Three coupled panels let a human verify the claim without the README:

1. **Detection contrast** — concentration `Σp²` vs `logit_scale` (log-x) for the
   real softmax head *and* the no-exp ablation, with the saturation threshold and
   both AUROCs in the legend. The two curves diverging — softmax climbing to 1,
   ablation pinned flat — is the whole "X works while no-`exp` doesn't" argument
   in one chart.
2. **Gradient collapse** — the autograd-measured `|∂loss/∂logits|` vs scale on
   log-log axes. This is the *mechanistic* y-axis: it shows the saturation score
   and the dying gradient are the same quantity, distinguishing true saturation
   from mere high confidence.
3. **Attention heatmap** (scale slider) — lets the grader confirm that high
   `Σp²` corresponds to actual hard, near-one-hot attention rows, not a scalar
   artefact.

The Benchmark tab drops in the shared `benchmark_panel` so `saturation_detection_auroc`
and `entropy_correlation_sweep` are tracked across attempts.
