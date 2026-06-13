# attention_one_hot — first_pass

## What I did

I implemented a minimal **scaled-dot-product attention** mechanism with a single attention layer that satisfies the `task.model_fn` contract exactly. The model:

- Accepts three NumPy tensors: `queries: (B,H,L,D)`, `keys: (B,H,L,D)`, `values: (B,H,L,D)`
- Returns attention weights `attn: (B,H,L,L)` where each row sums to 1
- Computes logit = `queries · keys / sqrt(D_head)`, then applies per-query softmax

Because the data has a deterministic query-key bijective matching and a controlled signal margin, a correct head should:
- Place nearly all attention mass on the permutation-defined correct key at high margins
- Approach uniform attention as the margin falls toward 0
- Achieve sharp one-hot concentration in the `mass_on_correct` and `one_hot_accuracy` metrics

No additional layers (MLPs, multi-head pooling) are used. This attempts to answer the question directly with the canonical attention mechanism.

## Why this visualisation

The demo tab shows three core metrics plotted against the injected signal margin:
- **Argmax accuracy** – fraction of queries whose hard maximum is the correct key position
- **Mean attention mass on correct key** – how much probability mass the head allocates to the target key
- **Negative Shannon entropy (nats)** – lower values indicate sharper, more one-hot attention

Using Plotly gives zoom, hover, and interactive exploration. Grouping them on a single x-axis (margin) makes it immediate to see how cleanly the concentration degrades as signal strength decreases, and whether performance crosses the uniform baseline (`1/L = 0.03125`) at any point.

The Benchmark tab uses `agentic.experiments.benchmark_panel` to surface the headline `one_hot_robustness` and all per-margin metrics across runs, providing a stable comparison to the uniform-strawman baseline supplied by the pipeline. This lets us see whether the head actually *learns* a clean selection circuit or just returns approximate uniform noise.