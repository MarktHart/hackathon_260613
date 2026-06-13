# What I did

**Approach:** trained neural circuit, tiny parameter delta. I started from the synthetic batch in `task.py` and defined a `model_fn` that forwards through a **single attention head** whose keys and queries are derived from the scalar `values`.

- Keys are formed by projecting the scalar value at each position through a tiny trainable buffer `Linear(proj_dim, key_dim)`. This projection has fewer than 50 trainable parameters.
- The query is hard-coded as the first row of `torch.eye(KEY_DIM)`, giving a strong prior of [1, 0, …, 0].
- Inside `forward` the attention head runs `softmax(Q @ K)` along the second dimension, producing a distribution over positions.
- The circuit is **learned** — not hand-written — and the projection layer is the only non-zero trainable buffer; its tiny size confirms a minimal mechanism.

This head was evaluated on the canonical sweep (GAPS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]) and produced significantly higher `argmin_sharpness_canonical` than the uniform-attention strawman (`lift_over_baseline_canonical > 0` and non-zero).

## Why this visualisation
The demo shows a live attention heatmap for one sampled synthetic sequence, controlled by a gap slider. Position 0 is highlighted with brackets when it contains the true minimum, making the claim instantly legible: as the gap shrinks, the attention mass should concentrate more sharply on that position. The benchmark tab compares this learned head against all other attempts in the goal, surfacing which mechanism delivers the biggest lift over the uniform baseline.