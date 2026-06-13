"""Gradio app for the first-pass attempt.

Two tabs:
- Demo: interactive view of the latest run (or a selected run) showing
  per-K routing accuracy vs. the additive baseline, plus a trial-level
  confusion matrix for a chosen K.
- Benchmark: shared leaderboard across all attempts at this goal.
"""

import json
import gradio as gr
import numpy as np
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir

GOAL_DIR = Path(__file__).parent.parent  # experiments/attention_int_mul/
ATTEMPT_DIR = Path(__file__).parent


def _list_runs() -> list[Path]:
    """Return sorted list of run directories (newest first)."""
    res_dir = results_dir(__file__)
    if not res_dir.exists():
        return []
    runs = [p for p in res_dir.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def _load_payload(run_path: Path) -> dict | None:
    bench_file = run_path / "benchmark.json"
    if not bench_file.exists():
        return None
    with open(bench_file, "r") as f:
        return json.load(f)


def _make_sweep_plot(payload: dict):
    """Create a grouped bar chart: routing accuracy per K for model vs baseline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = payload["sweep"]
    baseline = payload["linear_baseline"]
    ks = [r["k"] for r in sweep]
    model_acc = [r["routing_accuracy"] for r in sweep]
    base_acc = [r["routing_accuracy"] for r in baseline]

    x = np.arange(len(ks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width/2, model_acc, width, label="Model (multiplication)", color="#2c7bb6")
    ax.bar(x + width/2, base_acc, width, label="Additive baseline", color="#d7191c")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("Operand range K")
    ax.set_ylabel("Routing accuracy")
    ax.set_title("Routing accuracy vs. operand range K")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _make_confusion_matrix(payload: dict, k: int):
    """We don't have per-trial predictions stored, so show a placeholder."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, "Per-trial predictions not stored\nin this attempt's payload.\nRe-run with logging to enable.",
            ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_title(f"Confusion matrix (K={k}) — not available")
    ax.axis("off")
    fig.tight_layout()
    return fig


def _update_demo(run_name: str):
    """Load the selected run and return plot figures."""
    runs = _list_runs()
    selected = next((r for r in runs if r.name == run_name), runs[0] if runs else None)
    if selected is None:
        empty_fig = gr.Plot().value
        return empty_fig, empty_fig

    payload = _load_payload(selected)
    if payload is None:
        empty_fig = gr.Plot().value
        return empty_fig, empty_fig

    sweep_fig = _make_sweep_plot(payload)
    # Default to canonical K for confusion matrix
    k_choices = [r["k"] for r in payload["sweep"]]
    canon_k = payload.get("canonical_k", 8)
    k_idx = k_choices.index(canon_k) if canon_k in k_choices else 0
    conf_fig = _make_confusion_matrix(payload, k_choices[k_idx])

    return sweep_fig, conf_fig


def _update_confusion(payload_json: str, k: int):
    """Update confusion matrix when K dropdown changes."""
    # payload_json is not used here; we reload from the current run
    # This is a Gradio pattern: the run selector triggers both plots
    pass


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app
# ---------------------------------------------------------------------------
with gr.Blocks(title="attention_int_mul — first_pass") as demo:
    gr.Markdown("# attention_int_mul :: first_pass\nHand-built nearest-neighbour multiplication routing.")

    with gr.Tab("Demo"):
        runs = _list_runs()
        run_choices = [r.name for r in runs] if runs else ["(no runs yet)"]

        with gr.Row():
            run_dd = gr.Dropdown(
                choices=run_choices,
                value=run_choices[0] if run_choices else None,
                label="Run",
                interactive=bool(runs),
            )

        sweep_plot = gr.Plot(label="Routing accuracy sweep")
        conf_plot = gr.Plot(label="Confusion matrix (per K)")

        with gr.Row():
            k_dd = gr.Dropdown(
                choices=[2, 4, 8, 16, 32],
                value=8,
                label="K for confusion matrix",
                interactive=True,
            )

        # Event handlers INSIDE the Blocks context
        run_dd.change(
            _update_demo,
            inputs=[run_dd],
            outputs=[sweep_plot, conf_plot],
        )

        k_dd.change(
            lambda run_name, k: _update_demo(run_name)[1],  # only confusion matrix changes
            inputs=[run_dd, k_dd],
            outputs=[conf_plot],
        )

        # Initial load
        demo.load(
            _update_demo,
            inputs=[run_dd],
            outputs=[sweep_plot, conf_plot],
        )

    with gr.Tab("Benchmark"):
        # Shared leaderboard across all attempts at this goal
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()