import gradio as gr
import numpy as np
from agentic.experiments import benchmark_panel, latest_run

import sys
sys.path.append(".")

from experiments.attention_dot_product.task import (
    CANONICAL_D_HEAD,
    D_HEAD_SWEEP,
    generate,
    SEED,
    SEQ_LEN,
    NUM_SAMPLES,
)  # type: ignore[module-not-found]
from experiments.attention_dot_product.benchmark import score

demo: gr.Blocks

# Demo tab (visualiser that runs a new slice)
with gr.Blocks() as demo:
    # UI to pick a head dimension and display the reconstructed scores for
    # one sample.
    dd_d_head = gr.Dropdown(
        choices=D_HEAD_SWEEP,
        value=CANONICAL_D_HEAD,
        label="Head dimension d_head",
        interactive=True,
    )
    with gr.Row():
        gr.HTML("Predicted scores (1 sample) →")
        pred_heatmap = gr.Image(type="numpy", elem_id="heatmap")
    gr.Markdown("---")

    # Benchmark tab (history of all attempts under this goal)
    with gr.Blocks():
        benchmark_panel("experiments/attention_dot_product")

    # Run button that rebuilds the heatmap when the dropdown value changes
    btn_compute = gr.Button("Compute", variant="primary")

    def _compute(d_head: int):
        batch = generate(SEED)[D_HEAD_SWEEP.index(d_head)][0]  # pick first sample
        # Rebuild the model from main.py logic
        raw = np.matmul(batch.Q, np.transpose(batch.K, (0, 2, 1)))
        scores = raw / np.sqrt(d_head)
        # Gradio expects a 2D array for heatmap, so take the first sample
        return scores[0]

    # UI wiring: button triggers compute on every change
    btn_compute.click(
        fn=_compute,
        inputs=dd_d_head,
        outputs=pred_heatmap,
        queue=True,
    )

    # Start with the compute for canonical d_head
    demo.launch()