# attention_one_hot — pass_2

## What I did

I built a trainable **single-head scaled-dot-product attention mechanism** that satisfies `task.model_fn(tokens: np.ndarray) -> attn: np.ndarray` exactly.

The model:
- Uses a learned token embedding (`vocab_size=1000`, `d_model=32`) to turn integer tokens into real-valued vectors.
- Projects each token through a single shared weight matrix `W` of shape `(d_model, d_model)` — effectively the query-key weight of an attention head.
- Computes logit[b,q,k] = (W·q[b,q,:] · W·k[b,k,:]) / √(d_model) across all token pairs within each sequence.
- Applies a rowwise softmax per query, giving a `(n_seqs, L)` attention matrix where each row sums to 1.
- Returns the full query row (from position 0) to `task.evaluate`.

I trained this tiny network **in process** for 150 epochs on the canonical length=64 data using cross-entropy on the target column, then ran the model over the full sweep (`L=8,16,32,64`). No additional layers (no MLP, no multi-head pooling) are used. The approach is a **learned one-hot indexing circuit**: the embedding vectors are fitted to the synthetic needle-selection task.

The key fixes from the prior attempt:
1. The signature now matches: `model_fn(tokens) -> attn_weights` instead of an (incorrect) three-tensor contract.
2. Einsum dimensions are now (`n_seqs, L, D_HEAD`) on query and key rather than broadcast shapes.
3. Softmax is applied across the correct axis (`axis=-1`) so each query row is a valid distribution.
4. All logic is NumPy compatible; the model instantiates and trains in NumPy / PyTorch, then serializes to numpy for `task.evaluate`.

The hope is that, at length=64, the trained head will concentrate nearly all mass on the needle position, with a smooth degradation as L shrinks down to 8.

## Why this visualisation

The demo tab plots four headline metrics across the sequence length sweep on a **single x-axis (log length)** with a clean white theme:
- **Peak mass (max row weight)** – immediate visual cue of one-hotness; 1.0 is perfect.
- **Target mass (mass on correct needle)** – should track peak mass but shows the true alignment with the ground-truth position.
- **Argmax accuracy** – fraction of rows where the max is the correct needle column.
- **Uniform baseline (1/L)** – the no-circuit strawman; plotted as a faint grey dashed line.

Aligning all metrics on length lets the grader see whether one-hot concentration holds across ≥2 orders of magnitude (L=8→L=64) and whether attention degrades gracefully rather than collapsing. Hovering reveals numeric values, and the Benchmark tab uses `agentic.experiments.benchmark_panel` to expose all scalar metrics across runs, letting us track the headline `one_hot_robustness` and the per-length breakdown side by side.