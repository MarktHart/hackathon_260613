import gradio as gr
from agentic.experiments import load_task, benchmark_panel, results_dir

# Import the same model_fn used by main.py so the Demo tab matches the evaluated run.
def attention_argmin_model_fn(values: np.ndarray) -> np.ndarray:
    B, L = values.shape
    logits = -values
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    attn = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    return attn


with gr.Blocks() as demo:
    gr.Markdown("# Attention Argmin Demo")
    gr.Markdown(
        "This demo visualises a hand-coded \"attention argmin\" model — a single softmax "
        "over negated values that learns to place highest weight on the minimum position."
    )

    # Interactive demo: let the user pick the gap and see a sampled row.
    with gr.Row():
        gap_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.5,
            label="Minimum gap (lower value) vs base distribution",
        )
        visualize_btn = gr.Button("Visualize attention on a single sequence")
    with gr.Row():
        sequence_box = gr.Label(label="Sequence values (16 tokens; true min is highlighted with brackets)")
        attn_chart = gr.LinePlot(label="Attention weights over positions")

    def _vis(gap):
        # Build a small single-example batch just to get one sequence
        rng = np.random.default_rng()
        base = rng.uniform(0.0, 1.0, size=(1, 16))
        min_pos = rng.integers(0, 16, size=1)
        values = base.copy()
        values[0, min_pos] -= gap
        values += rng.normal(0.0, 0.05, size=values.shape)

        raw_seq = values[0].tolist()
        raw_seq[min_pos] = f"[{raw_seq[min_pos]:.3f}]"

        attn = attention_argmin_model_fn(values)
        positions = np.arange(16)
        return {
            "sequence_box": raw_seq,
            "attn_chart": {
                "x": positions,
                "y": attn[0].tolist(),
                "title": f"Attention weights (minimum = {values[0, min_pos]:.3f} at index {min_pos})",
                "xlabel": "Position",
                "ylabel": "Weight"
            }
        }
    visualize_btn.click(_vis, inputs=gap_slider, outputs=[sequence_box, attn_chart])

    gr.Markdown("---")
    gr.Markdown("## Benchmark History")
    # Scan the goal directory (../..) to show metric curves across all attempts.
    benchmark_panel("../../..", tab_name="Benchmark tab")

if __name__ == "__main__":
    demo.launch()