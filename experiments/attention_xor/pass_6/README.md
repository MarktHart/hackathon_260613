# attention_xor — pass_6

## What I did
**Hand-built (interp) attempt.** This is `base_model.py` reduced to its smallest
delta: a single self-attention head, **no MLP**, no positional encoding,
`d_model=4`, with hand-set token embeddings and identity Q/K/V (non-causal so
the CLS token at position 0 can read the later A/B tokens). The CLS query routes
attention equally onto the A and B tokens; each carries a **signed** value
feature `±1` for its bit, so the pooled CLS stream holds `x,y ∈ {±1}`. A
quadratic readout `(x−y)²−0.5` is positive exactly when `A≠B`, giving XOR at
accuracy 1.0 on every marginal (headline `xor_robustness` = 1.0). All compute
runs in torch on CUDA. `main.py` also records two ablation strawmen — zeroing
the attention output and swapping the quadratic readout for a linear one — both
of which collapse to the linear-probe floor.

## Why this visualisation
The Demo tab shows the full 4-row XOR truth table side-by-side with the two
ablations: the `full pred` column matches XOR on all four corners while the
`no-attn` and `linear` columns do not — making it visually obvious that *both*
the attention routing and the non-linear readout are load-bearing. The
"ablation accuracies" panel reads the artefact `main.py` wrote and contrasts
each knocked-out circuit's `p=0.5` accuracy against the linear floor, which is
the causal claim the goal cares about. The Benchmark tab drops in the shared
`benchmark_panel` so lift-over-linear and `xor_robustness` are comparable across
attempts.
