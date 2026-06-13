# attention_dtw / first_pass

## What I did

**Hand-built** (no training). The task hides a known monotone warp:
`keys[n] = queries[align[n]] + small noise`. So the alignment is recoverable
from **content alone** — each key should attend to the query it most resembles,
ignoring position entirely. The model function is `base_model.py`'s
self-attention with two minimal deltas: RoPE is removed (absolute position is
exactly the cue a real alignment circuit must *not* use), and the QK dot-score
is replaced by a negative-squared-L2 distance kernel,
`attn[n, m] = softmax_m(-‖k_n − q_m‖² / T)`, which is a cleaner content match
when query norms vary. I expose three heads so the demo can contrast the
mechanism with strawmen: head 0 = L2 content (the circuit), head 1 = raw
dot-product content (base_model-style), head 2 = a fixed diagonal positional
head. `evaluate` keeps the best head (head 0). On the canonical condition the
content head scores **best_head_overlap = 1.000** and holds it across the whole
warp sweep (**alignment_robustness = 1.0**), while the diagonal baseline decays
1.00 → 0.41 — direct evidence this is an alignment circuit, not a diagonal
shortcut. All compute runs in torch on CUDA.

## Why this visualisation

Two views answer two questions. The **heatmaps** (one per head, key index on y,
query index on x) with the ground-truth warp path overlaid in green let you
*see* that the content head's bright argmax band bends along the true warp while
the diagonal head's band stays a straight line — so as you slide the warp
dropdown, the content head tracks and the diagonal head visibly peels away from
the path. The **overlap-vs-warp line chart** is the quantitative headline: it
puts path overlap on the y-axis against warp on the x-axis, with the content
head, the mean-over-heads, the diagonal baseline, and chance (1/M) on the same
axes — a flat green line at 1.0 over a falling red diagonal line is exactly the
"retained under heavy warp" claim the benchmark's `alignment_robustness` metric
measures.
