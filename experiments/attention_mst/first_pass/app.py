import gradio as gr
import numpy as np
from agentic.experiments import benchmark_panel
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as patches

import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------
# Helper to render an MST edge set as a 1D bar chart, where each
# edge is a horizontal segment between its two node indices.
# --------------------------------------------------------------
def _render_mst(edge_list):
    # edge_list is list of (u, v, w) triples
    fig, ax = plt.subplots(figsize=(12, 1.5))
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlim(-1, 25)

    # Background of full possible connections (H-1 = 23)
    bg = np.zeros((23, 1))
    for i in range(len(bg)):
        bg[i, 0] = 0.2
    ax.imshow(bg, aspect='auto', cmap='gray', extent=[0, 24, -0.5, 0.5], alpha=0.3)

    # Highlight recovered edges in blue
    recovered = np.zeros((23, 1))
    for i, (u, v, _) in enumerate(edge_list):
        recovered[i, 0] = 1.0
    ax.imshow(recovered, aspect='auto', cmap='Blues', extent=[0, 24, -0.5, 0.5], alpha=1.0)

    ax.set_title('Recovered MST edges (blue = ground truth)')
    ax.set_xlabel('Head (node) index')
    ax.set_ylabel('Edge index')
    ax.grid(False)
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------
# Demo tab: interactive plot of predicted vs ground-truth MST edges
# --------------------------------------------------------------
with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                with gr.Column():
                    sigma_slider = gr.Slider(value=0.2, minimum=0, maximum=0.8, step=0.01,
                                             label="Noise sigma (used to draw observation batch)")
                    show_gt_chk = gr.Checkbox(value=True, label="Show ground-truth MST")
                    show_pred_chk = gr.Checkbox(value=True, label="Show predicted MST")
            with gr.Tabs():
                with gr.TabItem("Edge Set Plot"):
                    gt_plot = gr.Image(label="Ground-truth MST edge set")
                    pred_plot = gr.Image(label="Predicted MST edge set")

            def _demo_callback(sigma, show_gt, show_pred):
                # In a real hand-built attempt we'd be able to recompute the
                # prediction for any sigma by redrawing the observation batch.
                # Here we hardcode a single canonical observation batch (H=24, M=64)
                # and compute the prediction once at the start.
                # Because we are in a demo view we are not allowed to call
                # `task.evaluate()` repeatedly (that would run many GPU passes).
                # Instead we precomputed the payload and expose its slice here.
                # For a proper attempt you would recompute inside this callback,
                # drawing a fresh batch and running model_fn(sigma=batch_observations).

                # Simulate a fresh batch observation at current sigma
                # (use the fixed seed from the task to match the real payload's batch)
                from experiments.attention_mst.task import generate
                batch = generate(seed=7)   # canonical seed
                sigma_idx = batch.sigmas.index(sigma)

                # Extract predicted distance matrix from the precomputed payload (H=24)
                # Replace with real prediction if you want it to update.
                payload_path = "payload.json"  # would be saved under results/<timestamp>/
                if not os.path.exists(payload_path):
                    raise FileNotFoundError(f"Run main.py to generate {payload_path}")
                import json
                with open(payload_path, "r") as f:
                    payload_data = json.load(f)

                # For demo we only need the slice at current sigma
                sweep = next(s for s in payload_data["sweep"] if float(s["sigma"]) == sigma)
                pred_edges = [ tuple(int(x) for x in e.split(",")) for e in sweep["gt_mst_edges"] ]   # hack: parse string representation

                # Ground-truth edges are in the batch
                gt_edges = batch.gt_mst_edges

                # Render both as bar images
                gt_img = _render_mst(gt_edges)
                pred_img = _render_mst(pred_edges)

                return [gt_img, pred_img]

            show_gt_chk.input(_demo_callback, [sigma_slider, show_gt_chk, show_pred_chk], [gt_plot, pred_plot])
            sigma_slider.input(_demo_callback, [sigma_slider, show_gt_chk, show_pred_chk], [gt_plot, pred_plot])
            show_pred_chk.input(_demo_callback, [sigma_slider, show_gt_chk, show_pred_chk], [gt_plot, pred_plot])

        with gr.TabItem("Benchmark"):
            # Auto-populated leaderboard
            from agentic.experiments import benchmark_panel
            goal_dir = "experiments/attention_mst"
            metrics_panel = benchmark_panel(goal_dir, run_dirs=[results_dir(__file__)])
            metrics_panel.render()

if __name__ == "__main__":
    demo.launch()