# attention_graph_color / first_pass

## What I did

- Chose a **hand_built** circuit: no training, no learned weights. The model is a singleattention-like transform built entirely from hand-coded projections on `cuda` to satisfy the GPU guard (the pipeline verifies CUDA memory allocation, so even the static mechanism must run inside torch tensors). The signature `model_fn(adj, feats)` is satisfied by casting inputs directly to GPU tensors, doing the math there, then sending the result back to CPU.

- Designed a quadratic attention formula: `S = Q @ K.T`, where `Q` and `K` are derived from the colour part of `feats`. I used a hand-coded (k × k) projector that returns 1 for different colour combinations and 0 for identical ones, jittered slightly to avoid exact zeros (but the proper colouring guarantees no same-colour edges exist, so the projector’s self-entry never actually sees an edge). Then I masked this matrix with the adjacency matrix `adj` (placing most mass on edges) and normalised rows so the attention matrix is well-formed (rows of isolated nodes stay zero).

- The key insight is that because the projector explicitly encodes colour difference, the resulting attention matrix automatically favours differently-coloured pairs and especially differently-coloured edges; the adjacency matrix then reinforces that preference by zeroing out all non-edges, so all mass lands on edges and the `cross_edge_same_color` statistic is forced to zero as a sanity invariant.

- The hand-set weights are a (k, k) projector that I wrote by hand in NumPy, then lifted to `cpu` then to `cuda` in `model_fn`. Because it is a small, fixed matrix, it meets the hard-coded-weights bonus while being the minimal possible delta from a plain attention head (essentially a one-line QK product with a fixedcolour-matching matrix).

- The mechanism passes the GPU guard because every tensor (`qt`, `kt`, `S`, `mask`) lives on `cuda`. The NumPy-to-CUDA to-CPU round-trip is exactly the pattern the pipeline requires: `task.evaluate` gives NumPy, I put it on GPU to do the work, then I bring the answer back as NumPy.

- I then run `task.evaluate(model_fn)` inside `main.py`, which calls the `load_task` helper to get the data generator. The payload produced satisfies the exactly specified dictionary contract (see `task.py`). Finally, I write the payload + metrics into `benchmark.json` under the run directory managed by `results_dir(__file__)`.

- In `app.py`, I expose a Gradio demo with two tabs: a Demo tab that shows a Markdown summary of the headline metrics for the canonical (n=40, p=0.2) slice, and a Benchmark tab that drops in the canonical dashboard (`ageneric.experiments.benchmark_panel`) to show cross-attempt comparison. All Gradio event handlers live inside the `with gr.Blocks()` block to pass the boot-check.

## Why this visualisation

- A **single bar chart** of `color_separation_canonical` (headline metric) and `lift_over_linear_baseline` (separation gain over the uniform baseline) plus the per-graph summary table is enough to convey whether the mechanism encodes colour distinction. The uniform baseline gives zero separation, so any positive lift is a clear signal that the coloring circuit is active.

- I avoid heatmaps of the full (n × n) attention matrix because it is noisy at n=40 and adds little interpretability beyond the mean statistics the evaluator already reports. The dashboard on the Benchmark tab shows the same metrics across all attempts, so a simple per-graph summary plus the leaderboard plot makes the cross-strategy comparison visible without clutter.

- The Demo tab’s Markdown summary maps directly to the metric table in `benchmark.py`, so a human reviewer can read the same numbers in both places and know the story is consistent. The Benchmark tab is the pipeline-provided dashboard, so the visualisation is lean and the focus stays on the mechanism, not on the UI.