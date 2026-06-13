## What I did

This attempt implements a **hand-built attention circuit** (not learned, not trained) that directly encodes the local syntactic dependency structure the generator uses: every token attends to its immediate predecessor (shift = -1). The attention pattern is a deterministic sub-diagonal band of width 1, constructed as a torch tensor on CUDA and expanded to the required (B, H, T, T) shape. This is a `hand_built` attempt — the smallest possible delta from a random baseline, expressing the exact inductive bias the task tests for.

## Why this visualisation

The Demo tab plots the three sweep metrics (`mean_max_attn_to_target`, `mean_entropy`, `frac_peak_on_target`) across all five shifts (-2, -1, 0, +1, +2) as a bar chart. The canonical shift (-1) is highlighted in blue; distractors are light blue. Horizontal lines show the uniform-attention baseline (1/(T-1) ≈ 0.032) and the random-peak baseline (1/T ≈ 0.031). This layout makes the claim immediately legible: the hand-built head should hit ~1.0 on the canonical shift and ~0 everywhere else, with entropy near 0 and peak fraction near 1 — a sharp, falsifiable signature that the circuit matches the data-generating process.