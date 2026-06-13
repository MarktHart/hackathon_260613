# What I did

**Hand-built** boundary-detector circuit, expressed as genuine dot-product
attention running on the GPU (a small delta from `experiments/base_model.py`:
same `softmax(Q·Kᵀ)` mechanism, but with a hand-set 2-D Q/K projection instead
of a learned one). Unlike the previous attempt — which hard-coded the attention
*output* matrix — here the attention is actually computed from features derived
from the tokens. For each position I detect the delimiter (`argmax` of
`token == delim_id`), compute a **segment-sign** feature `s_i = sign(pos_i −
delim_pos) ∈ {−1, 0, +1}`, and a **special-token** indicator for the delimiter
and EOS. The head score is `Q_i·K_j = α·s_i·s_j − λ·special_j`, so every content
query concentrates its softmax mass *within its own segment* (`+α` same side,
`−α` across the boundary, `−λ` on delim/EOS). All four heads use this mechanism
at increasing strengths `α = {3,5,8,12}`, giving headline boundary sharpness
≈ 1.0 versus the uniform baseline's 0.0. **Faithfulness:** ablating the
segment-sign feature (`α→0`) collapses attention to uniform-over-content and
sharpness drops to ≈ 0, identical to the linear baseline — causal proof the
boundary behaviour comes from that feature, not from the special-token penalty.

# Why this visualisation

- **Per-head sharpness bar** puts the scored quantity itself
  (`within − max(delim, cross, eos)`) on the y-axis for both segA and segB
  queries, with the ablated head and the uniform baseline drawn as reference
  lines — so "the heads beat the strawman" is a one-glance comparison, exactly
  the benchmark's headline and lift metrics.
- **Attention heatmap** shows the two-block-diagonal structure with the DELIM
  and EOS columns marked; a faithful boundary head must light up only its own
  segment block, which the eye can verify directly.
- **Operating-range curve** sweeps the feature strength `α` across four orders
  of magnitude, showing where the mechanism turns on and that it saturates —
  the goal's relevant scaling axis.
- **Ablation bar** is the causal check: full circuit vs segment-feature-off vs
  uniform baseline, making it obvious the behaviour is driven by the
  segment-sign feature.
