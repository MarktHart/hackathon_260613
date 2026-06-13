# attention_dot_product / pass_3

## What I did
- Implemented a hand-built `model_fn` that computes scaled dot-product attention `softmax(QKᵀ/√d_head) · V` entirely on the CUDA device (`torch` + `DEVICE="cuda"`).
- The input and output match the task's NumPy API: inputs `(batch, n_heads, seq_len, d_head)` of shape `(8, 4, L, 16)` and outputs of the same shape.
- Weights are not learned; the circuit is the canonical dot-product formula with the scale `1/√16 = 0.25`.
- The function returns NumPy arrays; all intermediate tensors are on GPU.

## Why this visualisation
- The Demo tab shows a single-head `(32, 32)` attention score heatmap at a user-chosen sequence length, illustrating how softmax competition intensifies as length grows (the diagonal becomes sharper).
- The Benchmark tab aggregates MSE, relative error and cosine similarity across the entire length sweep (`[8, 16, 32, 64, 128]`) and compares against the baseline of uniform attention.
- The visualiser gives a concrete, interpretable slice of the mechanism while the leaderboard confirms that the implementation is indistinguishable from the reference at every length.