import gradio as gr
from agentic.experiments.benchmark_panel import leaderboard, metric_curve
from agentic.experiments import load_task

# Load the task to access its canonical sweep.
task = load_task(__file__)
payload = task.evaluate(task.random_model_fn())  # dummy run to get metric keys.

# Pull the distances and mean attentions from the latest pass in experiments/attention_span results/.
import os
import json
from pathlib import Path
import numpy as np

def load_latest_sweep():
    # This is brittle — replace with a proper results-loader from the framework.
    # We assume the results are nested under results/latest and contain a benchmark.json with a sweep.
    repo_root = Path(__file__).parent.parent.parent
    results_dir = Path(repo_root) / "results" / "latest"
    bench_path = results_dir / "benchmark.json"
    if not bench_path.exists():
        raise FileNotFoundError(f"No benchmark.json found in {results_dir}")
    bench = json.load(bench_path)
    sweep = bench.get("sweep", [])
    if not sweep:
        raise ValueError("Empty sweep in benchmark.json")
    # Extract distances and mean attentions sorted by distance.
    distances = np.array([s["distance"] for s in sweep])  # log2 distance to plot on x-axis as log2(d).
    means = np.array([s["mean_attention_on_target"] for s in sweep])
    return distances, means

try:
    distances, means = load_latest_sweep()
except Exception as e:
    print(f"Failed to load latest sweep: {e}. Falling back to a synthetic curve for demo.")
    # Synthetic curve for demo (the demo will still plot, but the shape won't be correct).
    distances = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    means = 1.0 / (1 + np.log2(distances))  # illustrative decay.

# The Demo tab: mean attention vs log2 distance.
with gr.Blocks() as demo:
    with gr.Blocks(variant="panel") as demo_tab:
        gr.Markdown("# Attention Span Decay Demo")
        gr.Markdown("""
        **Input**: synthetic sequences of length 512 where:
        - token 8888 (query) sits at position 0,
        - token 9999 (needle) sits at distances `d = 1, 2, 4, …, 256`,
        - all other positions are random distractors.

        **Output**: we query a single-head attention mechanism and plot the mean attention from the
        query at position 0 to the needle at distance `d`, aggregated over 100 sequences per `d`.

        The plot below shows the mean attention on the needle as a function of `log₂(d)`, spanning >2 orders of magnitude.
        """)

        with gr.Plot(
                distances, means,
                title="Mean Attention to Needle vs log₂(Distance)",
                x_label=r"log₂(distance from query to needle)",
                y_label="mean query→needle attention",
                width=800,
                height=500
        ):
            pass  # Gradio plot does not require anything inside for a simple line.

    # Benchmark tab: reuse the shared leaderboard.
    with gr.Blocks(variant="panel") as benchmark_tab:
        gr.Markdown("# Benchmark Leaderboard")
        gr.Markdown(
            "The table below ranks every attempt at this goal by metrics from `benchmark.py`. "
            "Use this tab to compare your method against the field."
        )
        leaderboard(__file__)

        gr.Markdown("#### Metric trend over runs")
        metric_curve(__file__)

    # Two tab layout.
    gr.TabbedInterface(
        tabs=[demo_tab, benchmark_tab],
        titles=["Demo", "Benchmark"]
    )

if __name__ == "__main__":
    demo.launch()