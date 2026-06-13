"""Gradio app for the hand-built NOT mechanism (pass_5).

Tabs:
- Demo: interactive sweep visualization showing how the NOT mechanism
  separates query→A attention across the superposition sweep.
- Benchmark: shared leaderboard across all attempts at this goal.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel

_GOAL_DIR = Path(__file__).parent.parent
_ATTEMPT_DIR = Path(__file__).parent
_RESULTS_DIR = _ATTEMPT_DIR / "results"


def _find_latest_run() -> Path | None:
    if not _RESULTS_DIR.exists():
        return None
    runs = sorted(_RESULTS_DIR.iterdir())
    return runs[-1] if runs else None


def _load_payload(run_dir: Path) -> dict | None:
    bench_path = run_dir / "benchmark.json"
    if not bench_path.exists():
        return None
    with bench_path.open("r") as f:
        return json.load(f)


def _plot_sweep(payload: dict) -> plt.Figure:
    sweep = payload["sweep"]
    baseline = payload["baseline"]
    cos_vals = [r["cos"] for r in sweep]
    sharp = [r["not_sharpness"] for r in sweep]
    base_sharp = [r["not_sharpness"] for r in baseline]
    gap = [r["suppression_gap"] for r in sweep]
    spec = [r["attend_specificity"] for r in sweep]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    # Top-left: Not sharpness vs baseline
    ax = axes[0, 0]
    ax.plot(cos_vals, sharp, "o-", label="Attempt (NOT mechanism)", color="#1f77b4", linewidth=2)
    ax.plot(cos_vals, base_sharp, "s--", label="Linear baseline", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("cos(k_B, k_A)")
    ax.set_ylabel("NOT sharpness")
    ax.set_ylim(0, 1.05)
    ax.set_title("NOT sharpness across superposition sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top-right: Suppression gap
    ax = axes[0, 1]
    ax.bar(cos_vals, gap, width=0.15, color="#2ca02c", edgecolor="k", label="Suppression gap")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("cos(k_B, k_A)")
    ax.set_ylabel("Suppression gap")
    ax.set_title("E[attn(A)|B=0] − E[attn(A)|B=1] (A=1)")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Attend specificity
    ax = axes[1, 0]
    ax.plot(cos_vals, spec, "^-", label="Attend specificity", color="#d62728", linewidth=2)
    ax.set_xlabel("cos(k_B, k_A)")
    ax.set_ylabel("Attend specificity (1 − false-attend rate)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Avoiding false attention to A when A absent")
    ax.grid(True, alpha=0.3)

    # Bottom-right: Summary table
    ax = axes[1, 1]
    robustness = payload.get("superposition_robustness", 0.0)
    ax.table(
        cellText=[
            ["NOT sharpness (cos=0.0)", f"{sharp[0]:.3f}"],
            ["Lift over baseline (cos=0.0)", f"{sharp[0] - base_sharp[0]:.3f}"],
            ["Superposition robustness", f"{robustness:.3f}"],
            ["Suppression gap (cos=0.0)", f"{gap[0]:.3f}"],
            ["Attend specificity (cos=0.0)", f"{spec[0]:.3f}"],
        ],
        colLabels=["Metric", "Value"],
        cellLoc="center",
        colWidths=[0.35, 0.15],
        bbox=[0.1, 0.1, 0.8, 0.8]
    )
    ax.axis("off")

    return fig


def _refresh_demo():
    run_dir = _find_latest_run()
    if run_dir is None or not run_dir.exists():
        return None, "No runs found."
    record = _load_payload(run_dir)
    if record is None:
        return None, "No benchmark.json in run."
    fig = _plot_sweep(record["payload"])
    return fig, f"Loaded {run_dir.name}"


with gr.Blocks(title="attention_not — pass_5") as demo:
    gr.Markdown("## attention_not — pass_5: Hand-built NOT via direct Q/K construction in head space")

    with gr.Tab("Demo"):
        plot = gr.Plot()
        status = gr.Markdown()
        refresh_btn = gr.Button("Refresh results")
        refresh_btn.click(_refresh_demo, outputs=[plot, status])
        demo.load(_refresh_demo, outputs=[plot, status])

    with gr.Tab("Benchmark"):
        benchmark_panel(_GOAL_DIR)


if __name__ == "__main__":
    demo.launch()