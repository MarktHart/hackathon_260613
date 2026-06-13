import gradio as gr
import importlib.util
import os
from agentic.experiments import benchmark_panel

# Load result directory from most recent run
def get_result_dir(goal_dir):
    # Scans goal_dir/attempt/*/results/<UTC>/benchmark.json
    attempt_dirs = [d for d in os.listdir(goal_dir) if d.startswith("attempt_") or d.startswith("first_pass") or d.startswith("trained_")]
    latest = None
    max_time = "0000/00/00"
    for attempt in attempt_dirs:
        attempt_path = os.path.join(goal_dir, attempt)
        if os.path.isdir(attempt_path):
            result_dirs = sorted([(os.path.dirname(d), d) for d in os.listdir(attempt_path) if d.startswith("results_")], reverse=True)
            for result_dir, full_name in result_dirs:
                time_str = full_name[8:16] + "/" + full_name[16:20] + "/" + full_name[20:24]
                if time_str > max_time:
                    max_time = time_str
                    latest = os.path.join(attempt_path, result_dir)
    if latest is None:
        raise FileNotFoundError(f"No runs found under {goal_dir}")
    return latest

# Get goal directory from __file__
current_dir = os.path.dirname(__file__)
goal_dir = os.path.dirname(os.path.dirname(current_dir))
run_dir = get_result_dir(goal_dir)

# Load benchmark results
def _load(payload_path):
    data = gr.load(payload_path)
    if not isinstance(data, dict) or "sweep" not in data:
        raise ValueError(payload_path + " lacks valid sweep")
    return data

with gr.Blocks() as demo:
    # Demo tab: simple control to show edge scores as heatmap
    gr.Markdown("# Sparse Causal Circuit (SCC) Demo")
    with gr.Row():
        with gr.Column():
            noise_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.1, value=0.3,
                label="Noise level applied to residual stream"
            )
            noise_slider.change(
                fn=lambda x: {"noise_std": x}, inputs=noise_slider, outputs=gradio.Output()
            )
        with gr.Column():
            edge_heatmap = gr.Heatmap(
                value=None, label="Edge scores across all head pairs (higher = stronger causal suggestion)",
                show_colorbar=True
            )
    # Load data on page load
    demo.load(
        _load,
        inputs=[],
        outputs=[edge_heatmap]
    )
    # Benchmark tab: standard leaderboard
    gr.Tabs()
    with gr.TabItem("Benchmark Panel"):
        gr.Markdown("## Benchmark Results")
        with gr.Blocks():
            gr.Mermaid(
                code=benchmark_panel(goal_dir),
                output_type=gr.Mermaid.OUTPUT_TYPE.SVG
            )

if __name__ == "__main__":
    demo.launch()