# attention_histogram / pass_3

## What I did

This is a **trained** attempt. The mechanism is a minimal **two-attention-block**
circuit — the smallest delta from `base_model.py` that solves the goal — whose
parameters are *discovered by gradient descent* rather than hand-set (the change
from pass_2, which hand-set an iterative power-iteration). Block 1 is a
**denoising attention** that writes to the residual stream:
`q' = q + alpha·(Knᵀ·softmax(beta1·Kn·q))`, pulling the noise-corrupted query
toward the key centroid; block 2 is the scoring readout `logits = beta2·(Kn·q')`.
Because the target direction is uniformly random every example, the data is
rotation-equivariant, so every learnable Q/K weight matrix collapses to a scalar
— I expose exactly the three scalars `(alpha, beta1, beta2)` such a circuit can
use and train them by cross-entropy on batches whose seeds are disjoint from the
eval seed. The denoising works because distractors sit at cosine `≈sim²` to each
other but `sim` to the target, making the **target the most central key**, so the
centroid points at it. Faithfulness is causal, not assumed: training discovers
`alpha>0`, and the **`alpha=0` knockout** removes block 1 while keeping the
trained temperature `beta2` — it stays sharp but its aim collapses back onto the
dot-product baseline, isolating the denoising block as the cause of correct
targeting.

## Why this visualisation

Three coupled views let a human check the claim. **(1) Histograms** put the
trained head, the `alpha=0` knockout, and the dot-product baseline side by side
at one similarity slice with the correct key in red — you see directly that the
mechanism both concentrates mass *and* moves the peak onto the right key, while
the knockout is sharp-but-misaimed. **(2) The sweep** plots sharpness *and*
hit-rate against rising distractor↔target cosine (the goal's interference axis),
with the ablation overlaid: the hit-rate panel's gap between the mechanism curve
and the ablation/baseline curves *is* the denoising block's causal effect, and
the chance line separates "sharp and right" from "sharp but wrong" (the trap the
headline `histogram_robustness` alone can't catch). **(3) Depth + training**
shows mean hit-rate vs number of denoising blocks (operating range — one block
captures the benefit) and the cross-entropy loss curve plus the learned scalars,
so the reader can confirm the mechanism was actually learned.
