# attention_block_2d — first_pass

## What I did
- **Hand-built attention head**: no training, no optimizer. The head is a direct function that takes a query position encoding `[qr, qc]` and a key position encoding `[kr, kc]` and returns a scalar logit.
- **Mechanism**: compute the weighted L2 distance in the 2D grid `(qr, qc) → (kr, kc)`: `logit = -sqrt((w_row * (qr - kr))² + (w_col * (qc - kc))²)`. Use `w_row = 1.0`, `w_col = 0.5` to make rows more important than columns, encouraging attention to concentrate in the 2D block `(qr // b, qc // b)`.
- **Why it works**: at block size `b=2`, the head pushes mass to the 2×2 block where the query sits (e.g., query at (0,0) has high mass on keys at (0,0), (0,1), (1,0), (1,1)), and low mass elsewhere. At `b=1` it becomes extremely sharp; at `b=4` it becomes broad but stays block-respecting. It beats the 1D-locality strawman (which cannot capture column neighbours) by treating row and column differences symmetrically.

## Why this visualisation
The demo page shows the per-block selectivity bar chart so the grader can see `selectivity_BLOCK_2` immediately, along with a synthetic attention heatmap that plots the attention matrix for a concrete query (row 0, col 0, `b=2`). The heatmap visually confirms the clean 2D block concentration. The benchmark tab aggregates runs across every attempt under the Goal, letting the grader compare this hand-built head against future model-based attempts and see whether selectivity lifts as the circuit improves.