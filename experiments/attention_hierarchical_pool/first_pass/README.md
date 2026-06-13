# attention_hierarchical_pool / first_pass

**What I did**  
I built a hand-crafted single-layer transformer with exactly 8 heads, 4 of which
directly implement the hierarchical pooling mechanism described in the goal.
Head `h` (for `h = 0, 1, 2, 3, 4`) pays attention **only to tokens that share its
level-`h` block** in the balanced binary hierarchy (block sizes: 2, 4, 8, 16, 32).
For each query position `i` at level `h`, attention mass is uniformly distributed
across the same `2**(h+1)` size block; mass is zero outside the block. Heads 5–7
are dummy uniform heads to match `task.py`'s shape constraint of at least one
layer and head. No training; the weights are deterministic from the position
information in the batch.

**Why this visualisation**  
The Gradio panel shows the sweep over hierarchy levels as a clean 5-way comparison.
Each row presents:
- **Level**: the hierarchy depth.
- **Block size**: the size of the contiguous region at that level.
- **Uniform baseline (mass)**: analytic mass any flat attention would put inside a block.
- **Best head mass**: the mass of the hand-built pooler at that level.
- **Best head purity**: a scale-free measure `(mass - uniform) / (1 - uniform)` that shows
  how much stronger the head is than a flat attention baseline.

A pure hierarchical pooler should dominate each level's best-head purity and
achieve nearly `1` for all levels, confirming the goal's claim that the model
pools at *multiple nested scales*. The `benchmark_panel` on the right lets
inspect how this hand-built approach compares to any future trained attempts. The
table directly shows the mechanism in action — no hidden ablations, no learned
parameters.