## What I did

This is a **hand_built** attempt — no training, no oracle. The mechanism is the
smallest possible delta from `base_model.py`'s attention block: a single
scaled-dot-product attention head with a temperature knob,
`attn = softmax(GAIN · scale · Q·Kᵀ)`, computed in torch on `cuda`. The only
per-condition signal is `scale`; `GAIN = 6.0` is one fixed global constant
standing in for the magnitude a trained Q/K projection would learn (random unit
vectors have tiny dot products, so a real head scales them up). Heavier-tail
conditions get a higher temperature → a sharper softmax → a few keys dominate →
a large quantile ratio (q90/q50); lighter-tail conditions get a low temperature
→ flatter softmax → ratio near the uniform baseline of 1.0. There is no MLP and
exactly one attention layer, which is sufficient. To establish **faithfulness /
causal evidence**, `main.py` runs two ablations through the identical
`task.evaluate` path: freezing the temperature to 1.0 (`no_temperature`)
collapses the pareto-vs-exponential lift to ~1.0, and replacing the softmax with
ReLU-normalisation (`linear_no_exp`) flattens the tail toward uniform — showing
the temperature-scaled `exp` softmax is the component responsible for the heavy
tail, beating both the uniform strawman and the ablated circuits.

## Why this visualisation

Three charts, each checking a distinct claim. The **sweep bar chart** puts
quantile ratio (q90/q50) on the y-axis with a dashed line at the uniform
baseline (=1.0); pareto bars (red) clearly stand above the baseline and the
exponential bars (blue) sit near it, which is exactly the heavy-vs-light split
the goal asks about. The **ablation chart** is the causal test: side-by-side
pareto/exp lift and canonical ratio for full vs temperature-ablated vs
linear-ablated vs uniform, so a human can see the structure vanish when the
mechanism is removed. The **Lorenz curve** makes "a few tokens dominate"
literal — cumulative attention mass vs fraction of (sorted) keys, with the
uniform diagonal for reference; the bowed gap is the heavy tail at a glance.
