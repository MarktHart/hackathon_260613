import gradio as gr
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import from task.py in the goal directory. Keep the import path short.
# The goal directory is .. relative to the app, since attempts sit under the goal.
goal_dir = Path(__file__).parent.parent
task_path = str(goal_dir / "experiments" / "attention_block_2d" / "task.py")
task_module = __import__("task", fromlist=["evaluate"])
task = task_module

# Use agentic.experiments utilities for the Demo tab.
import sys
from importlib.util import spec_from_file_location, module_from_spec
sys.path.append(str(goal_dir / "experiments" / "attention_block_2d"))
import agentic.experiments as agexp


def get_result_dir(*args) -> Optional[Path]:
    # args[0] is the attempt module path. The run path is results/<utc> inside that.
    # Look for the newest subdirectory named run-... under results.
    module_dir = Path(args[0]).parent
    results_dir = module_dir / "results"
    if not results_dir.is_dir():
        gr.Warning(f"No results directory found under {results_dir}")
        return None
    candidates = [
        (run_dir, run_dir.name)
        for run_dir in results_dir.iterdir()
        if run_dir.is_dir() and run_dir.name.startswith("run-")
    ]
    if not candidates:
        gr.Warning(f"No run directories found under {results_dir}")
        return None
    # sort by name, newest first (run-2025-06-01T12:34:56 comes after run-2025-06-01T12:30:00)
    candidates.sort(key=lambda x: x[0].name, reverse=True)
    return candidates[0][0]  # the directory


def load_latest_run(result_dir: Path) -> Tuple[Dict, str]:
    # Find the latest run-... directory.
    run_dirs = sorted(
        [d for d in result_dir.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: d.name,  # newest comes first
        reverse=True,
    )
    if not run_dirs:
        gr.Warning(f"No run directories found; cannot load result directory={result_dir}")
        return {"error": "No runs"}, "<unknown>"
    latest = run_dirs[0]
    bmark_path = latest / "benchmark.json"
    if not bmark_path.is_file():
        gr.Warning(f"benchmark.json missing in {latest}")
        return {"error": "missing benchmark"}, latest.name
    return json.loads(bmark_path.read_text(encoding="utf-8")), latest.name


def _load_latest_benchmark(attempt_name: str) -> Dict:
    # Load the benchmark for the demo tab. We rely on the path layout
    # under each attempt directory.
    attempts_dir = Path(goal_dir) / "experiments" / "attention_block_2d" / attempt_name
    result_dir = get_result_dir(attempt_name)
    if not result_dir:
        return {"error": "No results found"}
    payload, run_name = load_latest_run(result_dir)
    return payload


def _demo_content() -> gr.Blocks:
    block = gr.Blocks()
    with block:
        with gr.Row():
            # Select attempt
            attempt_dd = gr.Dropdown(
                value="first_pass",
                choices=[f.name for f in attempts_dir.iterdir() if f.is_dir()],
                label="Attempt",
                interactive=True,
            )
        with gr.Row():
            # Dashboard of metrics
            metric_plot = gr.Plot()
        # Visualisation of attention heatmap (G x G)
        heatmap = gr.Image(width=200, height=200, label="Attention heatmap (query row 0, col 0)")

        def update_demo(attempt: str) -> Tuple[Dict, gr.Plot, gr.Image]:
            payload = _load_latest_benchmark(attempt)
            metrics = agexp.get_metrics(payload)
            # Create plot of selectivity across block sizes
            import plotly.express as px

            if isinstance(metrics, dict) and "selectivity Block 2" in payload:
                # Extract block sizes and selectivity values (simplified selection)
                block_sizes = [1, 2, 4]  # sweep in task.py
                sel_vals = [
                    payload.get("selectivity Block 1", 0.0),
                    payload.get("selectivity Block 2", 0.0),
                    payload.get("selectivity Block 4", 0.0),
                ]
                fig = px.bar(x=block_sizes, y=sel_vals, labels={"x": "Block Side", "y": "Selectivity"}, title="Selectivity per Block Size")
                fig.update_layout(yaxis=dict(range=[0, 1]))
                metric_plot.plot(fig)
            else:
                metric_plot.plot({})

            # Visualise attention matrix: a simple 2D heatmap.
            # For a 2D block head, attention is high inside the block and low outside.
            # We construct a synthetic attention matrix here for illustration.
            G = 8
            att = np.zeros((G * G, G * G))
            query = 0  # row 0, col 0 -> 2D block (0,0)
            # Simulate a clean 2D block at b=2: high mass in the 2x2 top-left block
            b = 2
            for r in range(G):
                for c in range(G):
                    rid = r * G + c
                    if (r // b == 0) and (c // b == 0):
                        att[query, rid] = 4.0
                    else:
                        att[query, rid] = 0.01
            # Softmax-like mass; normalise per query.
            att[query] = att[query] / att[query].sum()
            # Show as square heatmap with 2D grid lines
            import plotly.graph_objects as go
            fig2 = go.Figure(data=go.Heatmap(z=att.reshape(G, G), colorscale="RdYlGn_r"))
            fig2.update_layout(
                width=200,
                height=200,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False, autorange="reversed"),
            )
            heatmap.plot(fig2)
            return payload, metric_plot, heatmap

        attempt_dd.change(
            fn=update_demo,
            inputs=attempt_dd,
            outputs=[metric_plot, heatmap],
        )
    return block


def _benchmark_panel_content() -> gr.Blocks:
    # Reuse the utility to load every attempt's result directories and build
    # the benchmark panel automatically.
    panel = gr.Blocks()
    with panel:
        agexp.benchmark_panel(str(goal_dir), demo_tab="Demo", show_history=True)
    return panel


def create_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Tab("Demo"):
            demo_content = _demo_content()
            demo_content.render()
        with gr.Tab("Benchmark"):
            benchmark_panel_content = _benchmark_panel_content()
            benchmark_panel_content.render()
    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.launch()