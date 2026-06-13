import gradio as gr
import numpy as np
from pathlib import Path
from agentic.experiments import get_run_dirs, benchmark_panel
from dataclasses import dataclass
from typing import Dict, Any

@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray

@dataclass(frozen=True)
class ModelOutput:
    attn_weights: np.ndarray
    values: np.ndarray

def demo_panel(payload_data: Dict[str, Any]) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                "### Identity Copy Head Demo\n"
                "This head copies the value vector **exactly at the same position** (diagonal attention),"
                "achieving perfect fidelity on uniform diagonal attention across all tokens."
            )
        with gr.Tabs():
            with gr.Tab("Headline Metrics"):
                with gr.Row():
                    gr.Label(label="Identity Fidelity (128)", value="{:.4f}".format(payload_data['identity_copy_fidelity_token_128']))
                    gr.Label(label="Linear Baseline", value="{:.4f}".format(payload_data['linear_baseline_fidelity_canonical']))
                    gr.Label(label="Lift", value="{:.4f}".format(payload_data['lift_over_linear_baseline']))
            with gr.Tab("Per-Token Sweep"):
                gr.Markdown("#### Copy Fidelity per sweep token\n| Token | Fidelity |\n|-------|----------|")
                for t in [0, 64, 128, 192, 255]:
                    gr.Markdown(f "| {t} | {payload_data[f'identity_copy_fidelity_token_{t}']:.4f} |")
                gr.Markdown("#### Diagonal Attention Mass\n| Token | Diag Mass |\n|-------|-----------|")
                for t in [0, 64, 128, 192, 255]:
                    gr.Markdown(f "| {t} | {payload_data[f'diag_attn_mass_token_{t}']:.4f} |")
            with gr.Tab("Sweep Plots"):
                tokens = [0, 64, 128, 192, 255]
                fids = [payload_data[f'identity_copy_fidelity_token_{t}'] for t in tokens]
                diag_m = [payload_data[f'diag_attn_mass_token_{t}'] for t in tokens]
                with gr.Row():
                    gr.LinePlot(
                        labels=["copy_fidelity"],
                        values=[fids],
                        x=tokens,
                        y=[0.0, 1.0],
                        title="Copy Fidelity across sweep tokens (best head)",
                        line_width=3,
                        fill='line'
                    )
                    gr.LinePlot(
                        labels=["diag_attn_mass"],
                        values=[diag_m],
                        x=tokens,
                        y=[0.0, 1.0],
                        title="Diagonal Attention Mass",
                        line_width=3,
                        fill='line'
                    )
        with gr.Row():
            run_btn = gr.Button("Show latest run")
            run_btn.click(fn=lambda: None, inputs=[], outputs=[])
    return demo

# Combined dashboard
goal_dir = Path(__file__).parent.parent
dashboard = benchmark_panel(goal_dir)

if __name__ == "__main__":
    app = gr.Blocks()
    with app:
        gr.Tabs().render()
        gr.Tabs().render()
    app.launch()