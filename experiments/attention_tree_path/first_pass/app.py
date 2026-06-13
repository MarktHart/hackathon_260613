import gradio as gr
import json
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent

def load_latest_run():
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None, "No results directory found"
    runs = sorted(results_dir.iterdir())
    if not runs:
        return None, "No runs found"
    latest = runs[-1]
    bench_file = latest / "benchmark.json"
    if not bench_file.exists():
        return None, f"No benchmark.json in {latest}"
    with open(bench_file) as f:
        data = json.load(f)
    return data, str(latest)

def load_run(run_name):
    results_dir = Path(__file__).parent / "results"
    bench_file = results_dir / run_name / "benchmark.json"
    if not bench_file.exists():
        return None, f"No benchmark.json in {run_name}"
    with open(bench_file) as f:
        data = json.load(f)
    return data, str(run_name)

def get_run_choices():
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []
    return sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)

def format_metrics(metrics_dict):
    if not metrics_dict:
        return "No metrics"
    lines = []
    for k, v in metrics_dict.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4f}")
        else:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)

def plot_sweep_bar(sweep_data):
    """Create a simple text-based bar chart for sweep results."""
    if not sweep_data:
        return "No sweep data"
    lines = ["Sweep Results (correct_attn_mean):", ""]
    for rec in sweep_data:
        depth = rec["depth"]
        rule = rec["path_rule"]
        mean = rec["correct_attn_mean"]
        std = rec["correct_attn_std"]
        n = rec["n_valid_queries"]
        bar = "█" * int(mean * 30)
        lines.append(f"  depth={depth}, rule={rule:20s}: {mean:.4f} ± {std:.4f} (n={n}) {bar}")
    return "\n".join(lines)

def plot_head_bar(head_slice):
    if not head_slice:
        return "No head data"
    lines = ["Per-Head Breakdown (canonical):", ""]
    for h in head_slice:
        mean = h["correct_attn_mean"]
        std = h["correct_attn_std"]
        bar = "█" * int(mean * 30)
        lines.append(f"  Head {h['head']}: {mean:.4f} ± {std:.4f} {bar}")
    return "\n".join(lines)

def update_demo(run_name):
    data, path = load_run(run_name)
    if data is None:
        return path, "", "", ""
    metrics = data.get("metrics", {})
    sweep = data.get("payload", {}).get("sweep", [])
    head_slice = data.get("payload", {}).get("head_slice", [])
    config = data.get("payload", {}).get("config", {})

    config_str = f"Run: {run_name}\n" + "\n".join(f"  {k}: {v}" for k, v in config.items())
    metrics_str = format_metrics(metrics)
    sweep_str = plot_sweep_bar(sweep)
    head_str = plot_head_bar(head_slice)

    return config_str, metrics_str, sweep_str, head_str

# Load initial run
initial_runs = get_run_choices()
initial_run = initial_runs[0] if initial_runs else None
init_config, init_metrics, init_sweep, init_head = "", "", "", ""
if initial_run:
    init_config, init_metrics, init_sweep, init_head = update_demo(initial_run)

with gr.Blocks(title="Attention Tree Path - First Pass") as demo:
    gr.Markdown("# Attention Tree Path: First Pass (Hand-Built Circuit)")
    gr.Markdown("Hand-built attention heads that trace tree paths using structural features (parent_ids, depths, is_leaf).")

    with gr.Row():
        with gr.Column(scale=1):
            run_dropdown = gr.Dropdown(
                choices=initial_runs,
                value=initial_run,
                label="Select Run",
                interactive=True
            )
            config_box = gr.Textbox(label="Config", value=init_config, lines=10, interactive=False)
        with gr.Column(scale=2):
            metrics_box = gr.Textbox(label="Benchmark Metrics", value=init_metrics, lines=12, interactive=False)

    with gr.Row():
        sweep_box = gr.Textbox(label="Sweep Results", value=init_sweep, lines=12, interactive=False)
        head_box = gr.Textbox(label="Per-Head Breakdown", value=init_head, lines=12, interactive=False)

    run_dropdown.change(
        fn=update_demo,
        inputs=[run_dropdown],
        outputs=[config_box, metrics_box, sweep_box, head_box]
    )

    # Benchmark tab - shared across all attempts
    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()