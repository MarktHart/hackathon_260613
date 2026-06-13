# What I did
This attempt implements a hand-built model function that computes exact softmax attention statistics on the GPU, exactly matching the `model_fn` contract required by `task.evaluate`. The function projects queries and keys using the provided head-specific `W_Q` and `W_K`, computes scaled dot-product attention with a GPU tensor, and extracts per-head metrics:

- **attn_entropy**: mean over query positions of row-wise Shannon entropy of attention distributions
- **attn_max**: mean over query positions of the maximum attention weight
- **attn_top1_frac**: mean fraction of attention mass allocated to the top-1 key position
- **attn_topk_frac**: mean fraction allocated to the top-4 key positions

All tensor operations occur on `cuda`, then results are detached and moved to CPU NumPy before being returned. Head labels are generated statically as `["head_0", "head_1", "head_2", "head_3"]`.

# Why this visualisation
We do not include a dynamic Demo tab, as the goal is a fully synthetic batched experiment with no interactive input. Instead, the Gradio app serves two purpose tabs:

1. **Benchmark** — `agentic.experiments.benchmark_panel` renders a leaderboard and metric curves that compare this attempt against others at `attention_sat`, tracking `saturation_robustness_L` and `saturation_robustness_alpha` across sweeps of `L` and `α`. This shows how consistently heads saturate under changing sequence statistics.

2. **Summary** — A concise markdown page explains the synthetic setup and the four per-head statistics. Future attempts need only produce these same arrays; the rest of the pipeline (evaluation, scoring, visual reporting) is already wired.

If later attempts produce per-head weight maps or attention matrices, we can extend the Demo tab to plot heatmaps per head. For now, the Benchmark tab is sufficient to answer the question: *which heads saturate robustly vs. spuriously*. The headline metric `saturation_robustness_L` (1 - CV of mean top1_frac across L) will tell us whether the method produces truly robust saturation or just one-shot high saturation at L=64.