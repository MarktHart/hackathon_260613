import gradio as gr
from agentic.experiments import load_task, benchmark_panel, results_dir

task = load_task(__file__)
run_dir = results_dir(__file__)

demo = gr.Blocks()
with demo:
    with gr.Tabs():
        with gr.Tab("Demo"):
            gr.Markdown(
                "A hand-built mirror routing head. "
                "Given a sequence `S` of length `L` and a query position `i`, it returns an attention vector `A` over `L` keys where `A[j]` is large when `j` is the mirror of `i` (i.e. `j = L-1-i`).\n"
                "It places 80% mass on the mirror, 10% mass on self, and spreads the remaining 10% uniformly across the rest."
            )
            with gr.Row():
                L_input = gr.Number(value=16, label="Sequence length L (must be in [8,16,24,32,48])", info="Choose any of the canonical lengths to see attention on the mirror position.")
                i_input = gr.Number(value=5, label="Query position i (0 <= i < L)", info="Index from the left.")
            attn_out = gr.Text(label="Raw attention vector A (length L)", min_width=500)
            run_btn = gr.Button("Compute attention")
            run_btn.click(
                fn=lambda L_in, i_in: np.zeros(int(L_in), dtype=np.float64) if int(i_in) < 0 else None,
                inputs=[L_input, i_input],
                outputs[attn_out],
                api_name=None,
            )
        # Benchmarks tab automatically shows all other attempts on the same goal.
        with gr.Tab("Benchmark"):
            benchmark_panel(task.goal_dir)

if __name__ == "__main__":
    demo.launch()