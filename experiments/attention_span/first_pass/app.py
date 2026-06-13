import gradio as gr
import json
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent


def load_latest_run():
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None, None
    runs = sorted(results_dir.iterdir(), key=lambda p: p.name, reverse=True)
    if not runs:
        return None, None
    latest = runs[0]
    benchmark_path = latest / "benchmark.json"
    if not benchmark_path.exists():
        return None, None
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    return benchmark, latest.name


def load_run(run_name):
    run_dir = Path(__file__).parent / "results" / run_name
    benchmark_path = run_dir / "benchmark.json"
    if not benchmark_path.exists():
        return None
    with open(benchmark_path) as f:
        return json.load(f)


def list_runs():
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []
    return sorted([p.name for p in results_dir.iterdir() if p.is_dir()], reverse=True)


def plot_attention_decay(benchmark):
    """Create a simple text-based plot data for the decay curve."""
    if not benchmark or "sweep" not in benchmark:
        return "No data"

    sweep = benchmark["sweep"]
    distances = [s["distance"] for s in sweep]
    means = [s["mean_attention"] for s in sweep]
    stds = [s["std_attention"] for s in sweep]

    lines = []
    lines.append(f"Model: {benchmark.get('model_name', 'unknown')}")
    lines.append(f"Half-life: {benchmark.get('attention_span_half_life', 'N/A')}")
    lines.append(f"Attention at canonical (d=128): {benchmark.get('attention_at_canonical', 'N/A'):.4f}")
    lines.append(f"Decay rate (λ): {benchmark.get('decay_rate', 'N/A'):.4f}")
    lines.append(f"AUC: {benchmark.get('area_under_curve', 'N/A'):.4f}")
    lines.append("")
    lines.append("Distance | Mean Attention | Std")
    lines.append("--------|----------------|------")
    for d, m, s in zip(distances, means, stds):
        lines.append(f"{d:>7} | {m:.6f}        | {s:.6f}")
    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown("# Attention Span — First Pass Demo")

    with gr.Row():
        run_dropdown = gr.Dropdown(
            choices=list_runs(),
            label="Select Run",
            value=list_runs()[0] if list_runs() else None,
            interactive=True,
        )
        refresh_btn = gr.Button("Refresh Runs")

    with gr.Tabs():
        with gr.TabItem("Decay Curve"):
            output_text = gr.Textbox(
                label="Attention to Key vs Distance",
                lines=20,
                max_lines=30,
                interactive=False,
            )

        with gr.TabItem("Raw Metrics"):
            metrics_json = gr.JSON(label="Benchmark Metrics")

    def update_display(run_name):
        if not run_name:
            return "No run selected", {}
        benchmark = load_run(run_name)
        if benchmark is None:
            return "Failed to load benchmark", {}
        return plot_attention_decay(benchmark), benchmark

    run_dropdown.change(update_display, inputs=run_dropdown, outputs=[output_text, metrics_json])
    refresh_btn.click(lambda: gr.Dropdown(choices=list_runs()), outputs=run_dropdown)

    # Initial load
    demo.load(
        lambda: update_display(list_runs()[0] if list_runs() else None),
        outputs=[output_text, metrics_json],
    )

    with gr.TabItem("Benchmark"):
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()