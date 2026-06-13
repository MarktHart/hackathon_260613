"""Gradio app: Demo tab (interactive graph viz) + Benchmark tab (leaderboard)."""
import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from agentic.experiments import benchmark_panel, load_task, results_dir


DEVICE = "cuda"


def model_fn(adjacency: np.ndarray) -> np.ndarray:
    """Same transitive-closure circuit as main.py, for live demo."""
    adj = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    n = adj.shape[0]
    eye = torch.eye(n, dtype=torch.float32, device=DEVICE)
    m = eye + adj
    m2 = m @ m
    m4 = m2 @ m2
    m5 = m4 @ m
    affinity = (m5 > 0).to(torch.float32)
    return affinity.detach().cpu().numpy()


def _latest_run_dir() -> Path:
    """Return the most recent run directory under results/."""
    base = results_dir(__file__).parent
    runs = sorted(base.glob("*"))
    return runs[-1] if runs else base


def _load_latest_payload() -> dict:
    """Load benchmark.json from the latest run."""
    run_dir = _latest_run_dir()
    bm_path = run_dir / "benchmark.json"
    if bm_path.exists():
        with open(bm_path) as f:
            return json.load(f)
    # Fallback: evaluate on the fly (slow but works for demo)
    task = load_task(__file__)
    return task.evaluate(model_fn)


def _viz_for_diameter(payload: dict, diameter: int) -> str:
    """Generate a simple text viz for a given diameter slice."""
    for rec in payload["sweep"]:
        if rec["diameter"] == diameter:
            m = rec["model"]
            b = rec["baseline"]
            def f1(c):
                tp, fp, fn = c["tp"], c["fp"], c["fn"]
                denom = 2 * tp + fp + fn
                return 2 * tp / denom if denom else 0.0
            return (
                f"Diameter {diameter}\n"
                f"  Model F1:     {f1(m):.4f}\n"
                f"  Baseline F1:  {f1(b):.4f}\n"
                f"  Lift:         {f1(m) - f1(b):+.4f}"
            )
    return "Slice not found"


with gr.Blocks() as demo:
    gr.Markdown("# attention_connected_components — first_pass\n"
                "Hand-built transitive closure via (I+A)⁵ on GPU.")

    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                diameter_dd = gr.Dropdown(
                    choices=[1, 2, 3, 5],
                    value=3,
                    label="Component diameter",
                    interactive=True,
                )
                run_btn = gr.Button("Evaluate slice", variant="primary")
            with gr.Column(scale=2):
                output_md = gr.Markdown()

        def on_run(diam):
            payload = _load_latest_payload()
            return _viz_for_diameter(payload, diam)

        run_btn.click(on_run, inputs=diameter_dd, outputs=output_md)
        demo.load(lambda: _viz_for_diameter(_load_latest_payload(), 3),
                  inputs=None, outputs=output_md)

    with gr.Tab("Benchmark"):
        # Goal directory is two levels up from this attempt's folder
        goal_dir = Path(__file__).parent.parent
        benchmark_panel(goal_dir)


if __name__ == "__main__":
    demo.launch()