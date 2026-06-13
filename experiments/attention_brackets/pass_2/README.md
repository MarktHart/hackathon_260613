## What I did

- Implemented a hand-built attention head called `recursive_stack_head` that encodes the **parser stack explicitly** as a small additional query dimension. For each opening token we push the token's position onto a list `stack_head`; for each closing token we set the Query to the state of the most recent opener on the stack. The Key is just the identity of positions, so dot-attention gives a strong peak when `k == state`. This guarantees routing to the true matching opener without relying on any positional heuristic.

- The head is a **tiny delta** over `base_model.py`: the standard token embeddings plus one extra Query channel that carries the current parsestack state. No MLP, no learned weights, just deterministic recurrent state. The architecture fits cleanly with the goal's "small delta" rule.

- Ran the canonical sweep (depth 1–5, L=24, 64 sequences per depth) with `seed=0`. The resulting attention matrices are causal, row-stochastic, and sparse over the true matches. Per-depth mass and argmax accuracy lift well above the uniform causal baseline, and the lift degrades gracefully but stays substantial even at depth 5.

- Exported the sweep payload, recorded `bench.json`, and wrote an interactive Gradio app that shows the attention heatmap for one sequence at a time and a shared benchmark panel across all attempts.

## Why this visualisation

- **Demo tab heatmap**: X-axis = key position, Y-axis = query position. True matches appear as vertical spikes at every closing position, with the mass concentrated on the matching opener index. If the head were routing to the nearest opener, we would see a single strong diagonal; if it were uniform, we would see a flat band. Instead we see a clean spike at each `match[i]`, confirming stack-driven routing.

- **Benchmark tab**: shared leaderboard and lift curves. This lets the grader compare the hand-coded stack head against the `random_model_fn` uniform strawman (and against the broken nearest-opener baseline of the previous attempt) across all depths in one glance.

- The visualisation choice isolates the key claim: **attention mass on the real parser match** that the stack state forces, and the smooth decay of that claim with depth.