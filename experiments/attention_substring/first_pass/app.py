import gradio as gr
from pathlib import Path
from agentic.experiments import benchmark_panel, results_dir, load_task
import json

def load_run(run_dir):
    if not Path(run_dir).exists():
        raise ValueError(f"Run directory not found at {run_dir}")
    with open<Path(run_dir) / "benchmark.json", "r") as f:
        return json.load(f)

def render_viz(run_dir):
    run = load_run(run_dir)
    if isinstance(run, dict) and "metrics" in run:
        metrics = run["metrics"]
        top_metric = list(metrics.keys())[0]
        value = metrics[top_metric]
        return f"Top metric `{top_metric}` = {value:.3f}"
    return "No metrics loaded."

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("Demo"):
            run_dd = gr.Dropdown(label="Run", choices=[run.name for run in results_dir(__file__).iterdir() if run.is_dir()])
            run_dd.change(render_viz, inputs=run_dd, outputs=gr.Markdown(label="Metrics"))
            run_dd.change(lambda run_dir: load_run(run_dir), inputs=run_dd, outputs=gr.JSON(label="Full payload"))

        with gr.TabItem("Benchmark"):
            benchmark_panel(__file__)

if __name__ == "__main__":
    demo.launch()