## What I did
This *hand-built* attempt implements a **KL-based classifier** for the five attention modes (`positional`,`uniform`,`diagonal`,`induction`,`previous_token`). It does not learn on the sweep; instead, it uses the exact clean-pattern matrices baked into `task.generate` as reference templates.

The classifier works per head:
- For each head’s `(L, L)` attention matrix, it compares every query row’s distribution to the clean distribution of each mode (using the `positional` matrix with anchor 0 as the positional mode, the `(i, i+1)` matrix for induction, etc.).
- It computes row-wise KL divergences (p from q), averages them per mode, then transforms the negative KL scores into a probability distribution over the five modes via a softmax on the GPU (`-KL` so lower divergence → higher score).
- All arithmetic happens on `cuda` inside PyTorch tensors with a single NumPy-to-torch / torch-to-NumPy boundary.

Because the clean matrices are hard-coded (they are the very patterns `task.generate` uses), this attempt exhibits the **hardcoded weights bonus**: it is written out in closed form without any learning step.

The classifier naturally handles noise — when a row is perturbed away from its mode’s ideal, the KL divergence to that mode rises and is reflected in lower predicted probability. No special robustness mechanism is built in; robustness emerges from the KL metric itself.

## Why this visualisation
The Demo tab shows a single attention pattern as a lower-triangular heatmap (L×L), its ground-truth mode, and the model’s per-mode probability bar (ordered positional → previous_token). The head index dropdown lets the grader browse all 50 heads from the canonical noise=0.0 sweep.

The KL-detail summary reports per-mode KL scores computed in the UI (for transparency), showing which clean template is farthest from the corrupted row. This visualisation makes the mechanism legible: the heatmap shows where mass lies (i.e., which pattern is being corrupted), the KL scores show *how much* each mode template deviates, and the bar chart shows the softmaxed verdict.

The Benchmark tab shows every attempt’s accuracy across the noise sweep, with the headline `mode_robustness` and per-mode canonical accuracy. High per-mode accuracy (especially on induction and positional modes, which have sharp off-diagonal spikes) is the strongest signal of faithfulness. The KL-based UI lets the grader sanity-check that the model is indeed reading *distributional similarity* to the clean matrices.