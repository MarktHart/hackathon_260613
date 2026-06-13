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

# Load the most recent run
def load_latest(pay: Dict[str, Any]) -> tuple[Dict[str, Any], str]:
    latest = pay[-1]
    run_id = Path(latest["run_dir"]).name
    return latest["payload"], f"run_{run_id}"

# Demo panel
def demo_panel(payload_data: Dict[str, Any]) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                "### Identity Copy Head Demo\n"
                "This head copies the value vector **exactly at the same position** (diagonal attention),"
                "achieving perfect fidelity on uniform diagonal attention."
            )
        with gr.Tabs():
            # Headline metrics
            with gr.Tab("Headline Metrics"):
                with gr.Row():
                    gr.Label(label="Identity Copies (best head fidelity)", value="{:.4f}".format(payload_data["身份复制"]['identity_copy_fidelity_canonical']))
                    gr.Label(label="Linear baseline", value="{:.4f}".format(payload_data["身份复制"]['linear_baseline_fidelity_canonical']))
		            gr.Label(label="Lift over baseline", value="{:.4f}".format(payload_data["身份复制"]['lift_over_linear_baseline']))
	            # Sweep per token
	            gr.Markdown("#### Fidelity per sweep token\n| Token | Fidelity |\n|-------|----------|")
	            for record in payload_data["sweep"]:
		            gr.Markdown(f"| {record['token']} | {record['copy_fidelity']:.4f} |")
	            gr.Markdown("#### Diagonal attn mass per sweep token\n| Token | Diag mass |\n|-------|-----------|")
	            for record in payload_data["sweep"]:
		            gr.Markdown(f"| {record['token']} | {record['diag_attn_mass']:.4f} |")
            # Sweep plots
            with gr.Tab("Sweep Plots"):
                tokens = [r['token'] for r in payload_data["sweep"]]
                fids = [r['copy_fidelity'] for r in payload_data["sweep"]]
                diag_m = [r['diag_attn_mass'] for r in payload_data["sweep"]]
                with gr.Row():
		            gr.LinePlot(
			            labels=["fidelity"],
			            values=[fids],
			            x=tokens,
			            y=[0.0, 1.0],
			            title="Copy Fidelity across sweep tokens",
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
        # Button to load latest run
        with gr.Row():
            run_btn = gr.Button("Show latest run")
            run_btn.click(fn=lambda: (None, None), inputs=[], outputs=[])
    return demo

# Combined dashboard
goal_dir = Path(__file__).parent.parent
dashboard = benchmark_panel(goal_dir)

def combined_app():
    demo = demo_panel({})
    with gr.Blocks() as combined:
        gr.Tabs().render()
        gr.Tabs().render()
    return combined

if __name__ == "__main__":
    combined_app().launch()