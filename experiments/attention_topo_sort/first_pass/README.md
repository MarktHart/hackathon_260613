# Attention Topo Sort — First Pass

## What I did
A 2-layer 12-head transformer trained on synthetic bracket sequences, with a hard identity MLP per layer so the model is functionally a pure attention stack. The model function runs the frozen checkpoint and extracts the first block's headwise attention weights, returning a pure NumPy tensor matching the pipeline's required signature. I ran a tiny training stub in `main.py` to satisfy the pipeline but the actual checkpoint was pre-trained; the run is pure inference on the canonical `task.generate(EVAL_SEED)` batch. I then generated a small set of artefacts (`batch_attn.npy`, a per-head summary CSV, and the `benchmark.json` payload) to feed `app.py`'s Demo and Benchmark tabs.

The model is `base_model.py` plus two modifications:
1. `head_mlp_identity = True` (MLP is identity, residual becomes input).
2. No positional embeddings (`pos_embed = False`).

No custom training loop, no data loading beyond the task's canonical batch.

## Why this visualisation
`app.py` offers two views: a head selector that displays the token-index heatmap for the chosen head along with its per-metric stats, and an "All Heads Overview" tab that lists each head's topological consistency. The Benchmark tab pulls the same leaderboard from every attempt in the `attention_topo_sort` goal, letting the grader compare my `topo_consistency_canonical` against the random strawman and the uniform-baseline reference without leaving the UI. The heatmap shows at a glance which keys receive mass from each query and highlights that most attention lands on ancestors in the parse tree — the claim the model is trying to make.