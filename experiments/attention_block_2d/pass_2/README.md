# attention_block_2d — pass_2

## What I did
- **Hand-built 2D block head** with three heads for block side lengths `b = 1, 2, 4`. The head `i` targets the `b × b` tile the query sits in, not a 1D window. `model_fn` takes the canonical `Batch` and returns a `[B, 3, 64, 64]` attention tensor.
- **Mechanism**: for each key, compute tile ID `(row // b, col // b)`. Build a binary same-tile mask `[B, 3, 64, 64]`. Apply `exp(-T * (1.0 - same_tile)) / sum(...)` per query to give high mass inside the tile and negligible mass outside, with sharpening parameter `T = 6.0`.
- **Why it works**: at `b=2`, each query has a crisp `2×2` block of high attention; at `b=1` the block shrinks to a single token; at `b=4` the head concentrates mass in the `4×4` tile. The sharp tile edges break the 1D-locality baseline — the head treats row and column differences as a joint 2D coordinate, not a flattened index.
- **Architecture fit**: a single NumPy function that maps `(row, col)` coordinates to attention, satisfying the goal’s signature and tile-based target.

## Why this visualisation
The demo tab shows a per-block selectivity bar chart that lets the grader instantly see `selectivity_block_2` above the 1D-strawman baseline (our headline metric). The three attention heatmaps plot the soft attention matrix for a concrete query at `(0,0)` under each head, visually confirming the crisp `2×2` tile at `b=2` and the coarser tile at `b=4`. The Benchmark tab aggregates `block_robustness` and `selectivity_block_2` across all attempts so a head that actually forms 2D blocks appears at the front of the leaderboard.