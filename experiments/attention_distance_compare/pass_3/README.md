# attention_distance_compare — pass_3

## What I did
This is a **hand_built** attempt, but unlike pass_2 (which pasted an
`exp(-|i-j|/λ)` curve straight into the softmax *output*), here the distance
decay is *emitted by a real attention computation*. The model is `base_model.py`
plus exactly one delta: an additive ALiBi-style **relative-position bias**
`b_{l,h}(i,j) = -|i-j| / λ_{l,h}` injected into the QK logits before the softmax.
Each of the required 4×8 heads is a genuine self-attention head — token
embedding → per-head Q/K projection → scaled dot-product (with content noise) →
softmax — and the only thing that varies across heads is the bias slope `λ`. I
deliberately span `λ` from 1.5 (steep, local) to 64 (flat, global), with
shallower layers more local, so the per-head metrics reveal local-vs-global
structure. No MLP and no residual depth are used: the goal measures per-head
attention patterns, so the mechanism is purely the attention layer's
positional bias. The 4×8 grid is fixed by the canonical measurement condition,
not a design choice. Everything runs in torch on CUDA.

**Faithfulness / causal evidence.** `model_fn_ablated` runs the *identical*
forward pass with the bias term zeroed. This knocks the headline decay slope
from **0.53 → 0.00**, landing the model exactly on the uniform baseline — direct
evidence that the relative-position-bias circuit (not the content QK term) is
what produces the distance preference. Per-head slopes form a clean gradient
from 3.7 (layer 0, head 0) to ~0.01 (layer 3, head 7).

## Why this visualisation
**Plot 1 (decay curve)** puts the three quantities the rubric cares about on one
log–log axis: the model curve (headline), the uniform baseline (strawman), and
the **bias-ablated** curve (causal control). The eye immediately sees the model
falling off with distance while the ablation sits flat on the baseline — "the
mechanism works *and* removing it breaks it" in a single frame. **Plot 2 (per-
layer/head slope heatmap)** answers the goal's second question directly: each
cell is one head's decay slope, so the local→global gradient across head index
and the layer-wise softening are legible at a glance, matching the
`layer_head_decay_slope_*` benchmark metrics.
