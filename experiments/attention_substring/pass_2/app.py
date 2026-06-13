import gradio as gr
from pathlib import Path
from agentic.experiments import benchmark_panel, results_dir, load_task
import json

def load_run(run_dir):
    if not Path(run_dir).exists():
        raise ValueError(f"Run directory not found at {run_dir}")
    with open(Path(run_dir) / "benchmark.json", "r") as f:
        return json.load(f)

def render_viz(run_dir):
    run = load_run(run_dir)
    if isinstance(run, dict) and "metrics" in run:
        metrics = run["metrics"]
        top_metric = "substring_detection_canonical"
        if top_metric in metrics:
            value = metrics[top_metric]
            return f"**{top_metric}** = {value:.3f}"
        else:
            return "No headline metric found."
    return "No metrics loaded."

def show_head_heatmap(json_data):
    # Simplify JSON for demonstration
    if isinstance(json_data, dict) and "attn_weights" in json_data:
        # We show attention from the target position to all positions
        # for the best head (head 0 in our case)
        attn = json_data["attn_weights"]  # [n_layers, n_heads, seq, seq]
        head = attn[0, 0]  # [seq, seq] for head 0
        return gr.DataFrame(head)
    return "No attention weights found."

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Demo"):
            run_dd = gr.Dropdown(label="Run", choices=[run.name for run in results_dir(__file__).iterdir() if run.is_dir()])
            run_dd.change(render_viz, inputs=run_dd, outputs=gr.Markdown(label="Metrics"))
            run_dd.change(load_run, inputs=run_dd, outputs=gr.JSON(label="Full payload"))
            attn_btn = gr.Button("Show Attention Heatmap (Head 0)")
            attn_btn.click(show_head_heatmap, inputs=gr.JSON(label="Full payload"), outputs=gr.DataFrame(label="Attention Weights"))
            token_acc = gr.Markdown(label="Token Prediction Accuracy (optional)")
        with gr.TabItem("Benchmark"):
            benchmark_panel(__file__)

if __name__ == "__main__":
    demo.launch()