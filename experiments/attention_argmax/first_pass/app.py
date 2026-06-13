import gradio as gr

# agentic imports.
from agentic.experiments import benchmark_panel, results_dir

# Goal-specific import from the sibling directory.
from experiments.attention_argmax.task import generate

# ---- Demo tab: visualize the generated batch and the attention vector ----
with gr.Blocks() as demo:
    # 1. Choose a run to load.
    with gr.Blocks():
        run_path = results_dir(__file__)  # latest run for this attempt.
        run_path = run_path.parent / run_path.name  # ensure it's a path, not a timestamp substring.
        run_path = gr.Textbox(label="Loaded Run Directory", value=str(run_path), interactive=False)

    # 2. Show a single deterministic example (seed = 0) at canonical L=128.
    with gr.Blocks():
        gr.Markdown("### Sample Sequence (L=128, seed=0)")
        seed_slider = gr.Slider(0, 99, value=0, label="Generation seed", step=1)
        batch_view = gr.Dataframe(label="Tokens (rows are sequences, columns are positions)",
                                  height=128)

    # 3. Show the attention vector of a chosen sequence and head.
    with gr.Blocks():
        seq_idx_slider = gr.Slider(0, 127, value=0, label="Sequence index")
        head_idx_slider = gr.Slider(0, 7, value=0, label="Attention head")
        attn_plot = gr.LinePlot(label="Attention probabilities over position",
                                x_label="position",
                                y_label="probability",
                                width=400,
                                height=200)

    # ---- Update the batch view when seed changes.
    with gr.Blocks():
        def _update_batch(seed: int):
            batch = generate(seed)
            # Build a list of lists for Gradio Dataframe (rows are sequences).
            rows = []
            for b in range(len(batch.tokens)):
                tokens = [str(t) for t in batch.tokens[b]]
                needle = batch.needle_pos[b]
                tokens[needle] = str(needle) + "*"
                rows.append(tokens)
            return rows

        seed_slider.change(
            fn=_update_batch,
            inputs=[seed_slider],
            outputs=batch_view
        )
        demo.load(_update_batch, [seed_slider], [batch_view])

    # ---- Update the attention plot when seq/head changes.
    with gr.Blocks():
        def _sample_attention(seq_idx: int, head_idx: int):
            # For the hand-built "first pass", we construct the attention vector on the fly.
            # A more realistic demo would load it from the recorded payload in result_dir.
            L = 128
            attn = np.full(L, 1e-8, dtype=np.float64)
            p = seq_idx  # placeholder "needle" for demonstration; not real.
            attn[p] = 1.0 - (L - 1) * 1e-8
            # Return a list of (x, y) for gr.LinePlot
            return list enumerate(attn)

        seq_idx_slider.change(
            fn=_sample_attention,
            inputs=[seq_idx_slider, head_idx_slider],
            outputs=attn_plot
        )
        head_idx_slider.change(
            fn=_sample_attention,
            inputs=[seq_idx_slider, head_idx_slider],
            outputs=attn_plot
        )
        demo.load(_sample_attention, [seq_idx_slider, head_idx_slider], [attn_plot])

    # ---- Benchmark tab: drop in the auto-generated leaderboard ----
    with gr.Blocks():
        with gr.Blocks():
            gr.Markdown("# Benchmark (Headline metric: argmax_robustness)")
        benchmark_panel(__file__)

if __name__ == "__main__":
    demo.launch()