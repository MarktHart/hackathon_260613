## What I did
This is a first-pass hand-built attempt that directly codes the A* heuristic into the attention head's logits. Given an 8×8 grid, current agent position, and goal position, the model function computes for each of the 9 positions in the 3×3 neighborhood:
- `center_offset = Manhattan distance from candidate to agent` (our `g` term),
- `manhattan_to_goal = Manhattan distance from candidate to goal` (our `h` term),
- combines them as `f = g + h`, and
- returns `-inf` for any obstacle (grid == 1). The agent's current position is masked out by the `evaluate` step, so we never place a token there.

All tensors are built on `cuda` and converted to NumPy only at the return boundary to satisfy the hard GPU-execution guard.

## Why this visualisation
The visualisation explains only the hand-coded mechanism, not a learned weight heatmap. The demo tab states the two components of the A* heuristic (`g` = center offset, `h` = Manhattan to goal) and their combination, plus the -inf clipping on obstacles. The Benchmark tab overlays all past runs for this goal so you can see how this hand-coded baseline ranks against trained and ablated attempts. No interactive demo is needed because the circuit is deterministic: for each fixed seed configuration, the model always produces the same attention logits. The key claim is architectural — that the A* heuristic is implementable by a single attention head with hand-set weights — and the visualisation makes that functional decomposition explicit.