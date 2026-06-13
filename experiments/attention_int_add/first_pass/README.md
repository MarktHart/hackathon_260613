# What I did

This attempt trains a small 3-layer transformer (d_model=128, 4 heads, d_ff=512) from scratch on the canonical 3-digit addition dataset. The model follows the `base_model.py` pattern: token + positional embeddings, causal self-attention blocks with residual connections and LayerNorm, and a tied output head. Training uses AdamW with cosine annealing for 50 epochs on the full `seed=0` batch (1200 problems across 4 carry buckets). The `model_fn` runs inference on GPU and returns logits for all positions; `task.evaluate` extracts predictions at the four SUM positions.

# Why this visualisation

The Demo tab lets the grader probe the model's carry handling interactively: they can type any pair of 3-digit numbers and see the predicted sum digits, the true sum, and a carry-count analysis. Four preset buttons target the critical regimes (0, 1, 2, 3 carries) so the grader can instantly verify whether the model generalises beyond the no-carry cases. The run dropdown shows per-slice exact-match rates and the headline `carry_robustness` metric, making the quantitative claim immediately visible alongside the qualitative demo.