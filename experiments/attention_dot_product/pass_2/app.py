import gradio as gr
import numpy as np
import json
from agentic.experiments import benchmark_panel, latest_run

# Import the task symbols we need (via relative import from the goal directory)
from experiments.attention_dot_product.task import (
    D_MODEL,
    N_HEADS,
    D_K,
    SEQ_LEN,
    generate,
    _true_scores,
)

demo: gr.Blocks

# Demo tab (visualiser for a selected input scale)
with gr.Blocks() as demo:
    gr.Markdown("# attention_dot_product / pass_2 Demo")
    with gr.Tabs():
        with gr.TabItem("Visualiser"):
            scale_slider = gr.Slider(
                label="Input scale (x)",
                minimum=0.5,
                maximum=8.0,
                value=1.0,
                step=0.5,
            )
            scale_btn = gr.Button("Recompute scores", variant="primary")
            # Output component to show a single head's heatmap for the selected scale
            vis_out = gr.Image(type="numpy", elem_id="heatmap")
            gr.Markdown("Tip: choose a scale > 2.0 to see how the mechanism handles larger dot products.")

        with gr.TabItem("Benchmark History"):
            benchmark_panel("experiments/attention_dot_product")

    # UI wiring: the button triggers a compute that rebuilds the heatmap
    # when the scale is changed.
    def _compute(scale: float):
        # Rebuild the full batch at canonical seed
        batch = generate(seed=0)
        # Interpolate to the chosen scale: take the sweep value >= chosen
        # scale and linearly blend with the next lower one (or just use the
        # nearest).
        nearest_idx = next(i for i, v in enumerate(batch.scales) if v >= scale)
        scale0 = batch.scales[nearest_idx - 1] if nearest_idx >= 1 else None
        scale1 = batch.scales[nearest_idx]
        X = batch.X_sweep[nearest_idx]
        if scale0 is not None and abs(scale0 - scale) < abs(scale1 - scale):
            idx0 = nearest_idx - 1
            idx1 = nearest_idx
            alpha = (scale - scale0) / (scale1 - scale0)
            X = X * alpha + batch.X_sweep[idx0] * (1 - alpha)

        # Hand the batch to the model function defined in main.py.
        S = model_fn(batch.W_Q, batch.W_K, batch.W_V, batch.W_O, X)

        # Pick a single head to visualise (head 0) and extract the (seq, seq) slice.
        # Gradio expects a 2D array for the heatmap.
        head_vis = S[:, :, 0]                     # (16, 16)
        return head_vis

    # Connect the button to the compute function on scale changes.
    scale_btn.click(
        fn=_compute,
        inputs=scale_slider,
        outputs=vis_out,
        queue=True,
    )

    # Start by visualising the canonical scale (1.0).
    demo.load(_compute, inputs=scale_slider, outputs=vis_out)

if __name__ == "__main__":
    demo.launch()