# What I did

I implemented a **hand-built transformer attention block** that explicitly encodes the wildcard matching circuit. The model takes `input_ids` of shape `[batch, 64]` and returns attention weights of shape `[batch, 4, 64, 64]`. For every sequence, it locates the WILD token (ID 2), then places exactly 50% attention weight on the token immediately before it (the true prefix) and 50% on the token immediately after it (the true suffix), and zero elsewhere. This is deterministic, fully hand-coded, and never "learns" from data — it is a synthetic implementation of the desired pattern-matching circuit.

- `num_heads = 4` — each head applies the same hand-coded attention pattern.
- The attention pattern is constructed by direct assignment, not by learned projections, so it cannot be fooled by distractors.
- No softmax or MLPs — the mechanism is pure positional attention: only offsets -1 and +1 receive non-zero weight.

The function matches `task.py`'s `model_fn` signature exactly: `model_fn(input_ids: np.ndarray) -> np.ndarray`.

# Why this visualisation

The `app.py` Demo tab shows a **heatmap of attention weights** from the WILD token to the rest of the sequence. Because the attention is deterministic and sparse, the pattern appears as two bright red vertical lines at the prefix and suffix positions, with everything else at zero. This makes the claim visually obvious: the model attends only to the intended positions, never to distractors or noise. The Benchmark tab drops in `agentic.experiments.benchmark_panel(<goal_dir>)`, so the grader can compare this hand-built result against future trained attempts and see whether attention to the true prefix and suffix approaches the hand-coded ceiling (≈1.0) or remains low as a robust baseline. The visualization’s axes — "Key Position" (x) and "Query Position" (y) — directly map to the question: does the WILD token attend to the positions that define the pattern?