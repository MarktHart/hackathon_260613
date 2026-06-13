# attention_dot_product / first_pass

## What I did
- Implemented a hand-built model function `model_fn(Q, K)` in NumPy that exactly evaluates the scaled dot-product formula: `scores = Q Kᵀ / √d_head`.
- The function takes the query and key tensors as provided by `task.py`, computes their matmul, normalises by `√d_head` (where `d_head` is the shape dimension of Q and K), and returns the `(num_samples, seq_len, seq_len)` score matrix.
- The attempt is deterministic, uses no neural network, and reproduces the ground-truth mechanism by literal computation — no learned parameters are involved.

## Why this visualisation
- The demo tab shows a single-slice view of the reconstructed scores as a heatmap, letting a human see the matrix structure at a chosen `d_head`.
- The Benchmark tab aggregates all attempts’ runs and plots metrics like `fidelity`, `cosine`, and `scale_accuracy` across the `d_head` sweep, directly answering whether the direction and magnitude are reproduced robustly.
- Because this is a hand-computation attempt and not a learned model, the heatmap and metric history together serve as a correctness check: the mechanism should be exact for every `d_head`, with `scale_accuracy → 1` as the only variable.