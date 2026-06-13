"""Gradio demo for the attention-shift-by-k first-pass attempt."""

import json
from pathlib import Path

import gradio as gr
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


def load_payload(run_dir: Path):
    """Load payload.json from a run directory."""
    payload_path = run_dir / "payload.json"
    if not payload_path.exists():
        return None
    with payload_path.open() as f:
        return json.load(f)


def make_demo_plot(summary: dict):
    """Create a matplotlib figure for the demo tab."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = summary["k"]
    target_mass = summary["mean_target_mass"]
    accuracy = summary["shift_accuracy"]
    baseline = summary["uniform_baseline_mass"]
    lift = summary["lift_over_uniform"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: target mass vs baseline
    x = np.arange(len(k))
    width = 0.35
    ax1.bar(x - width/2, target_mass, width, label="Target mass", color="#2ca02c")
    ax1.bar(x + width/2, baseline, width, label="Uniform baseline", color="#7f7f7f", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"k={ki}" for ki in k])
    ax1.set_ylabel("Softmax mass on target")
    ax1.set_title("Mean target mass across valid query positions")
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Right: shift accuracy and lift
    ax2.plot(k, accuracy, "o-", label="Shift accuracy (argmax hit rate)", color="#d62728")
    ax2.plot(k, lift, "s-", label="Lift over uniform", color="#1f77b4")
    ax2.set_xlabel("Offset k")
    ax2.set_ylabel("Score")
    ax2.set_title("Accuracy & lift vs offset")
    ax2.legend()
    ax2.set_ylim(0, 1.05)
    ax2.set_xscale("log", base=2)

    fig.tight_layout()
    return fig


def make_heatmap(payload: dict):
    """Create a heatmap of target mass per query position for each k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = payload["sweep"]
    seq_len = payload["config"]["seq_len"]

    # Build matrix: rows = k (in sweep order), cols = query position
    # Value = softmax mass on target (i-k) for that query position
    # We need to recompute per-position masses from the payload... 
    # But payload only has aggregates. Let's reconstruct from the model.
    # Actually, we can't without the model_fn. Let's just show the aggregate bar chart.
    # For a heatmap, we'd need per-position data. Skip for now.
    pass


with gr.Blocks() as demo:
    gr.Markdown("# Attention Shift by K — First Pass Demo")
    gr.Markdown(
        "Hand-built QK circuit with identity positional embeddings and a "
        "k-dependent shift matrix for keys. The mechanism places all attention "
        "mass exactly on position `i - k` for query position `i`."
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
            # Drop in the shared benchmark panel for this goal
            benchmark_panel(str(GOAL_DIR)).render()

    def update_demo(run_name: str):
        if not run_name:
            return None, "No runs found. Run `main.py` first."
        run_dir = RESULTS_DIR / run_name
        summary = load_summary(run_dir)
        payload = load_payload(run_dir)
        if summary is None or payload is None:
            return None, f"Missing summary/payload in {run_dir}"

        fig = make_demo_plot(summary)

        # Build metrics markdown
        sweep = payload["sweep"]
        canon = next(s for s in sweep if s["k"] == payload["config"]["canonical_k"])
        md = f"""### Metrics for run `{run_name}`

| Offset k | Valid positions | Target mass | Shift accuracy | Uniform baseline | Lift |
|----------|----------------|-------------|----------------|------------------|------|"""
        for s in sweep:
            md += f"\n| {s['k']} | {s['n_valid']} | {s['mean_target_mass']:.4f} | {s['shift_accuracy']:.4f} | {s['uniform_baseline_mass']:.4f} | {s['mean_target_mass'] - s['uniform_baseline_mass']:.4f} |"

        md += f"""

**Canonical (k=4) summary:**
- Target mass: **{canon['mean_target_mass']:.4f}**
- Shift accuracy: **{canon['shift_accuracy']:.4f}**
- Lift over uniform: **{canon['mean_target_mass'] - canon['uniform_baseline_mass']:.4f}**
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


if __name__ == "__main__":
    demo.launch()