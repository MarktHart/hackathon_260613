# What I did

This attempt implements a **hand-built mechanism** that directly mimics the Game of Life update for each cell. The model uses a single attention layer with a fixed convolutional pattern over the neighborhood and a hand-coded rule layer that applies the birth/survival logic. No training is performed; all weights are hand-designed.

Key choices:
- **QKV design** sets the query to the center cell and keys/queries to the eight surrounding neighbors, encoded with a learned relative offset bias.
- **Softmax denominator** forces a uniform attention weight across all neighbors, then each neighbor is counted by taking the dot product of its position embedding with the query's position embedding.
- **Rule head**: two parallel projections compute birth and survival logits, then a final gate mixes alive vs dead updates in one pass.

## Why this visualisation

`app.py` visualises the **logit update map** — predicted logits per cell alongside the true next-state board — so a human can instantly compare predictions against the ground truth. A colour heatmap makes birth/survival patterns legible without extra decoding. The Demo tab shows the model's output on a random initial board with three different densities (0.1, 0.3, 0.5) chosen interactively; the Benchmark tab plots the full density sweep as F1 vs density curves with confidence bands.

This visualisation directly answers the question: *does the architecture reproduce the discreteGame of Life update at every cell?* A glance reveals whether cells are correctly predicting neighbors and applying the rule, and the per-density curve shows robustness across densities.