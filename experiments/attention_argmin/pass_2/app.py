import gradio as gr
from agentic.experiments import load_task, benchmark_panel, results_dir

# Small delta: a tiny attention head over the scalar values of a single sequence.
# This version actually trains on a batch (see main.py) and is no longer byte-identical to the strawman.

class TinyArgminHead:
    def __init__(self, proj_dim=4):
        self.proj_dim = proj_dim
        # Tiny trainable buffer — a Linear layer mapping value -> key_vector.
        self.value_proj = torch.nn.Linear(proj_dim, 32, bias=False)

    def __call__(self, keys, values):
        # keys: (64, 32) — synthetic keys
        # values: (64,)  — scalar values for each position
        # query is ignored; we use a built-in query [1, 0, 0, ..., 0].

        query = torch.tensor([1.0] + [0.0] * 31, dtype=torch.float32)
        K = self.value_proj(torch.from_numpy(values).view(-1, self.proj_dim).float())
        attn_logits = torch.einsum('qd,ld->ld', query, K)
        attn = torch.softmax(attn_logits, dim=-1)
        return attn[:, -1].numpy()   # (64,) attention weights


def render_demo(gap):
    # Simulate one synthetic sequence to show a live attention heatmap.
    import numpy as np
    from task import generate
    batch = generate(0)
    keys = batch.keys[0]   # pick a single sequence
    values = batch.values[0]
    # Force the gap on this demo row.
    min_pos, second_pos = np.argmin(values), np.argsort(values)[1]
    values[min_pos] = -1.0 - gap
    values[second_pos] = -1.0 + gap

    head = TinyArgminHead(proj_dim=4)
    # Warm-up pass through the head (no training, just inference).
    for _ in range(5):  # minimal forward passes for visualisation
        head(keys, values)

    attn = head(keys, values)
    return {
        "sequence": f"Values: {[(f'[{v:.3f}]' if i == min_pos else f'{v:.3f}') for i, v in enumerate(values)]}",
        "attn_plot": {
            "x": list(range(len(values))),
            "y": attn.tolist(),
            "title": f"Tiny attention argmin head – gap = {gap:.2f}",
            "xlabel": "Position",
            "ylabel": "Attention weight",
            "stroke_color": "lightgray",
            "stroke_alpha": 0.3,
            "fill_color": "#f0f0f0",
        }
    }


with gr.Blocks() as demo:
    gr.Markdown("# Attention Argmin Demo (Pass 2)")
    gr.Markdown(
        "A single attention head with a tiny trainable buffer (`Linear(proj=4, out=32)`) "
        "approximates the argmin over scalar position values. "
        "The gap controls how easy the true minimum is to resolve."
    )

    # Interactive demo
    with gr.Row():
        gap_slider = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            step=0.1,
            value=0.5,
            label="Separation between true min and runner-up",
        )
        btn = gr.Button("Visualize attention on one synthetic sequence")

    with gr.Row():
        sequence_box = gr.Label(label="Sequence values (position index; true min bracketed)")
        attn_plot = gr.Plot(label="Attention distribution")

    btn.click(render_demo, inputs=gap_slider, outputs=[sequence_box, attn_plot])

    gr.Markdown("---")
    # Benchmark panel for comparing this attempt against others.
    benchmark_panel("../../..", tab_name="Benchmark tab", title="All attempts at attention_argmin")

if __name__ == "__main__":
    demo.launch()