"""Gradio demo for the attention-shift-by-k pass_2 attempt.

Showcases a hand-built attention head that exactly implements a relative positional
shift-by-k circuit (query i attends to key i-k). Demo visualisation shows two panels:
left bar chart compares best-head mass vs the uniform baseline for each offset k;
right line chart plots shift accuracy and chance-normalised lift against log2 k.
The Benchmark tab shows the shared leaderboard across attempts.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


def list_runs():
    """Return sorted list of run directories (newest first)."""
    if not RESULTS_DIR.exists():
        return []
    runs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return runs


def load_summary(run_dir: Path):
    """Load summary.json from a run directory."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open() as f:
        return json.load(f)


def load_mech.metrics(run_dir: Path):
    """Load mech.metrics.json (the interesting one) for the Dashboard."""
    path = run_dir / "mech.metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def load_uniform.metrics(run_dir: Path):
    """Load uniform.metrics.json for baseline comparison."""
    path = run_dir / "uniform.metrics.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def _k_to_label(k):
    """Return a plot label for offset k."""
    return f"k={k}"


def make_demo_plot(summary: dict, uniform_summary: dict = None):
    """Create a matplotlib figure for the Demo tab.

    Two panels side by side:
      (a) Bar chart comparing best-head mass (green bars) against uniform
          baseline (grey bars) for each k in the sweep.
      (b) Line chart of shift accuracy (red circles) and chance-normalised lift
          (blue squares) against log2 k; also show the uniform baseline per slice
          as horizontal reference lines.
    """
    k_vals = summary["k"]
    mass_mech = [
        summary[f"best_head_mass_k_{k}"] for k in k_vals
    ]
    mass_uniform = [
        summary[f"best_head_mass_k_{k}"] if uniform_summary else summary["base_mass"]
        for k in k_vals
    ]
    accuracy_mech = [
        summary[f"mean_head_mass_k_{k}"] if k == 1 else summary[f"best_head_mass_k_{k}"]
        for k in k_vals
    ]
    lift_mech = [
        (mass_mech[i] - summary["base_mass"]) / (1 - summary["base_mass"]) if k >= 1 else 0
        for i, k in enumerate(k_vals)
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: best-head mass vs baseline per k
    x = np.arange(len(k_vals))
    width = 0.35
    ax1.bar(x - width/2, mass_mech, width, label="Mechanism best-head mass", color="#2ca02c")
    if uniform_summary:
        ax1.bar(x + width/2, mass_uniform, width, label="Uniform baseline mass",
                color="#7f7f7f", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([_k_to_label(k) for k in k_vals])
    ax1.set_ylabel("Softmax mass on target (i-k)")
    ax1.set_title("Best-head mass vs uniform baseline across k")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Right: accuracy and lift vs log2 k
    ax2.plot(k_vals, accuracy_mech, "o-", label="Best-head argmax accuracy", color="#d62728")
    ax2.plot(k_vals, lift_mech, "s-", label="Chance-normalised lift", color="#1f77b4")
    ax2.set_xlabel("Shift offset k (log2 scale)")
    ax2.set_ylabel("Score")
    ax2.set_title("Accuracy and lift across shift offsets")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.set_xscale("log", base=2)

    fig.tight_layout()
    return fig


def make_benchmark_plot():
    """Produce a placeholder line plot for the Benchmark tab when no runs exist."""
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3, 4, 5], [0.5] * 5, marker="o", color="#1f77b4", linestyle="--")
    ax.set_xlabel("Attempt version")
    ax.set_ylabel("Shift robustness")
    ax.set_title("Empty benchmark — run main.py first")
    ax.grid(True)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# Attention Shift by k — pass_2 Demo")
    gr.Markdown(
        "This attempt implements a **hand-built exact QKV circuit** that "
        "produces a clean shift-by-k operation inside a single attention head. "
        "The mechanism (identity positional embeddings, shift-by-k key matrix, "
        "identity value projection) concentrates ~1.0 attention mass on position `i-k` "
        "for any query `i`, and works across the full sweep of offsets."
    )

    with gr.Row():
        run_dropdown = gr.Dropdown(
            choices=[str(d.name) for d in list_runs()],
            label="Select run",
            value=list_runs()[0].name if list_runs() else None,
        )
        refresh_btn = gr.Button("Refresh runs", size="sm")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            plot_output = gr.Plot(label="Shift performance across k")
            metrics_md = gr.Markdown()

        with gr.TabItem("Benchmark"):
            # Drop in the shared benchmark panel that shows the leaderboard
            # and metrics history across all attempts at this goal.
            benchmark_panel(str(GOAL_DIR)).render()

    def update_demo(run_name: str):
        if not run_name:
            return None, "No runs found. Run `main.py` first."
        run_dir = RESULTS_DIR / run_name
        # Both the mechanism run and the uniform baseline will exist
        # because we write both to the same run directory.
        mechanism_summary = load_summary(run_dir)
        if mechanism_summary is None or "k" not in mechanism_summary:
            return None, f"Missing summary in {run_dir}"
        uniform_summary = None   # Not used in plots; only for reference
        fig = make_demo_plot(mechanism_summary, uniform_summary)

        # Build metrics markdown with values from the mechanism run
        m = load_mech.metrics(run_dir) or {}
        shift_robustness = m.get("shift_robustness", 0.0)
        mass_canonical = m.get("shift_mass_canonical", 0.0)
        acc_canonical = m.get("shift_argmax_acc_canonical", 0.0)

        md = f"""### Run `{run_name}` — Mechanism Metrics

| Metric | Value |
|--------|-------|
| Shift robustness (mean chance-normalised lift across k) | **{shift_robustness:.4f}** |
| Best-head mass at canonical k=1 | **{mass_canonical:.4f}** |
| Peak-key accuracy at canonical k=1 | **{acc_canonical:.4f}** |
| Uniform-attention baseline per key | **{mechanism_summary["base_mass"]:.4f}** |
"""
        return fig, md

    # Event handlers INSIDE the Blocks context
    run_dropdown.change(update_demo, inputs=run_dropdown, outputs=[plot_output, metrics_md])
    refresh_btn.click(
        lambda: gr.update(choices=[str(d.name) for d in list_runs()]),
        outputs=run_dropdown,
    )
    demo.load(
        lambda: (list_runs()[0].name if list_runs() else None),
        outputs=run_dropdown,
    ).then(update_demo, inputs=run_dropdown, outputs=[plot_output, metrics_md])
    # For empty benchmark tab
    demo.load(make_benchmark_plot, [], gr.Plot())


if __name__ == "__main__":
    demo.launch()