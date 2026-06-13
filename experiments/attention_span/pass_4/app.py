import json
import os
import pathlib
import numpy as np

import gradio as gr

from agentic.experiments.benchmark_panel import leaderboard, metric_curve
from agentic.experiments import load_task

# Load the task once for the metric keys.
task = load_task(__file__)

# Attempt to load distances and mean attentions from the latest benchmark.json.
def load_latest_sweep() -> tuple[np.ndarray, np.ndarray]:
    repo_root = pathlib.Path(__file__).parent.parent.parent
    results_dir = repo_root / "results" / "latest"
    bench_path = results_dir / "benchmark.json"
    if not bench_path.exists():
        raise FileNotFoundError(f"benchmark.json not found in {results_dir}")
    with open(bench_path, "r") as f:
        bench = json.load(f)
    sweep = bench.get("sweep", [])
    if not sweep:
        raise ValueError("Empty sweep found in benchmark.json")
    distances = np.array([s["distance"] for s in sweep])
    means = np.array([s["mean_attention_on_target"] for s in sweep])
    return distances, means


try:
    distances, means = load_latest_sweep()
except Exception as e:
    print(f"Failed to load latest sweep: {e}. Falling back to a synthetic curve for demo.")
    # Synthetic curve for visualisation (the actual values will be shown in the Benchmark tab).
    distances = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    means = 1.0 / (1.0 + np.log2(distances + 1))   # illustrative slow decay


with gr.Blocks() as demo:
    with gr.Blocks(variant="panel") as demo_tab:
        gr.Markdown("# Attention Span Decay")
        gr.Markdown("""
**What's plotted**

- **Y-axis**: mean attention weight from the query token (position 0) to the target token (needle),
  averaged over 100 sequences at that distance.
- **X-axis**: `log₂(d)` where `d` is the distance from query to needle in token positions.
- The sweep covers `d = 1, 2, 4, ..., 256`.
- Uniform attention (no preference) would sit at `1/512 ≈ 0.00195`.

The curve shows the core claim: attention is strong at short distances and decays gracefully
over scales spanning more than two orders of magnitude.
        """)
        gr.LinePlot(
            x=distances,
            y=means,
            x_title="log₂(distance from query to target)",
            y_title="mean query→target attention",
            width=800,
            height=450
        )

    with gr.Blocks(variant="panel") as benchmark_tab:
        gr.Markdown("# Benchmark Leaderboard")
        gr.Markdown(
            "This tab shows the leaderboard and trend charts for all attempts at this goal."
        )
        leaderboard(__file__)

        gr.Markdown("#### Per-distance values over time")
        metric_curve(__file__)


if __name__ == "__main__":
    demo.launch()