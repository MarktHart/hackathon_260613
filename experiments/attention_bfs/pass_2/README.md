# attention_bfs — pass_2: Recurrent Attention for Multi-Hop BFS

## What I did
- **Mechanism**: Hand-coded *recurrent* attention that propagates reachability one hop per iteration. Each step computes `frontier @ adjacency` (query=frontier, key=adjacency, value=adjacency rows) to produce the next frontier, exactly mirroring BFS's queue expansion. After `h` steps, the accumulated reachable set matches ground-truth `≤h`-hop reachability.
- **Key difference from first_pass**: The previous attempt used a single attention pass with random weights, collapsing to a one-hop lookup. This attempt applies attention *recurrently* `h` times — the minimal architectural change needed for genuine multi-hop propagation.
- **No training, no learned weights**: The circuit is fully specified by the adjacency matrix itself. The "Q projection" is the identity (frontier vector), "K projection" is the adjacency matrix, "V projection" is the adjacency matrix — all hand-set, zero parameters.
- **GPU execution**: The iterative matrix multiplications run on CUDA via PyTorch, satisfying the pipeline's GPU requirement while staying numerically exact.

## Why this visualisation
- **Step-by-step propagation view**: The Demo tab shows each hop as a row — attention scores, new frontier, cumulative reachable, and graph highlighting. This makes the *mechanism* visible: you see the frontier expand outward like a wave, exactly as BFS does.
- **Ground-truth overlay**: Final prediction vs. true reachable set with precision/recall/F1 lets you verify correctness at a glance.
- **Interactive parameters**: Varying seed, `p`, and `h` demonstrates operating range — the mechanism works for any graph topology and hop budget because it *is* BFS implemented in attention primitives.
- **Benchmark tab**: Uses the shared `benchmark_panel` to show this attempt's F1 at `hops=5` (canonical) beating the 1-hop baseline, with robustness across the sweep — the quantitative proof that recurrent attention solves the multi-hop problem where single-shot attention fails.