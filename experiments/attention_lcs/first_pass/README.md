# What I did

This is a **hand-built** attempt (`attempt_type: hand_built`). I constructed four cross-attention heads manually, without training:
- **Head 0 (Exact Match)**: Attention weight = 1 where query token equals key token, 0 otherwise, then normalized. This directly implements the core LCS principle — matched positions share the same symbol.
- **Head 1 (Shifted Match)**: Attention on diagonals offset by ±1, capturing near-alignments.
- **Head 2 (Random)**: Random noise attention as a negative control.
- **Head 3 (Uniform)**: Uniform attention = random baseline.

The model runs on GPU (CUDA) as required. The hypothesis is that Head 0 should show significant LCS lift because LCS matches are a subset of exact token matches.

# Why this visualisation

The Demo tab shows a horizontal bar plot comparing **LCS Attention Mass** and **Lift over Baseline** across all four heads side by side. This makes it immediately visible whether the exact-match head (Head 0) outperforms the uniform baseline and the negative-control heads. The canonical metrics (best-head mass, lift, robustness) are shown as large numbers for quick reading. The run dropdown lets the grader inspect any historical run. The Benchmark tab embeds the shared leaderboard so this attempt's `lcs_lift_canonical` appears in context of future attempts.