"""Gradio app for the hand-built NOT mechanism (pass_3).

Tabs:
- Demo: interactive sweep visualization showing how the NOT mechanism
  suppresses the target key across the superposition sweep.
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
    with bench_path.open() as f:
        return json.load(f)


def _plot_sweep(payload: dict) -> plt.Figure:
    sweep = payload["sweep"]
    cos_vals = [r["cos"] for r in sweep]
    sharp = [r["negation_sharpness"] for r in sweep]
    base_sharp = [r["baseline_negation_sharpness"] for r in sweep]
    attn_abs = [r["target_attn_absent"] for r in sweep]
    attn_pres = [r["target_attn_present"] for r in sweep]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)

    # Top-left: Negation sharpness vs baseline
    ax = axes[0, 0]
    ax.plot(cos_vals, sharp, "o-", label="Attempt (NOT mechanism)", color="#1f77b4", linewidth=2)
    ax.plot(cos_vals, base_sharp, "s--", label="Linear baseline", color="#ff7f0e", linewidth=2)
    ax.set_xlabel("cos(k_neg, k_t)")
    ax.set_ylabel("Negation sharpness")
    ax.set_title("Target attention suppressed by marker")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Top-right: Target attention (absent vs present)
    ax = axes[0, 1]
    x = np.arange(len(cos_vals))
    width = 0.35
    ax.bar(x - width/2, attn_abs, width, label="Marker absent", color="#2ca02c", alpha=0.8)
    ax.bar(x + width/2, attn_pres, width, label="Marker present", color="#d62728", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{c:.1f}" for c in cos_vals])
    ax.set_xlabel("cos(k_neg, k_t)")
    ax.set_ylabel("Softmax attention to target")
    ax.set_title("Target attention with/without marker")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-left: Lift over baseline
    ax = axes[1, 0]
    lift = [s - b for s, b in zip(sharp, base_sharp)]
    colors = ["#2ca02c" if v > 0 else "#d62728" for v in lift]
    ax.bar(cos_vals, lift, width=0.15, color=colors, edgecolor="k", label="Lift over linear")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("cos(k_neg, k_t)")
    ax.set_ylabel("Sharpness lift")
    ax.set_title("Advantage over softmax competition alone")
    ax.grid(True, alpha=0.3, axis="y")

    # Bottom-right: Mechanism schematic (text summary)
    ax = axes[1, 1]
    ax.axis("off")
    canonical = payload.get("canonical_cos", 0.0)
    canon_idx = next(i for i, c in enumerate(cos_vals) if abs(c - canonical) < 1e-9)
    canon_sharp = sharp[canon_idx]
    canon_lift = lift[canon_idx]
    robustness = payload.get("superposition_robustness", 0.0)
    summary = (
        f"NOT Mechanism Summary\n"
        f"─────────────────────\n"
        f"Canonical (cos={canonical:.1f}) sharpness: {canon_sharp:.3f}\n"
        f"Canonical lift over baseline: {canon_lift:.3f}\n"
        f"Superposition robustness: {robustness:.3f}\n\n"
        f"Mechanism: Detect negation marker at keys[1] by its\n"
        f"projection onto the known neg_anchor (basis[8]).\n"
        f"Suppress target logit (slot 0) by SCALE × |k₁·neg_anchor|.\n"
        f"When marker absent: k₁ = absent_slot ⟂ neg_anchor → no suppression.\n"
        f"When marker present: k₁ = cos·k_t + sin·neg_anchor →\n"
        f"suppression ∝ sin = √(1−cos²).\n\n"
        f"This makes the *target logit itself drop* — genuine\n"
        f"content-specific inhibition, not just softmax competition."
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))

    return fig


def _refresh_demo(run_name: str | None):
    if run_name is None:
        run_dir = _find_latest_run()
    else:
        run_dir = _RESULTS_DIR / run_name
    if run_dir is None or not run_dir.exists():
        return None, gr.update(choices=[], value=None), "No runs found."
    payload = _load_payload(run_dir)
    if payload is None:
        return None, gr.update(choices=[], value=run_name), "No benchmark.json in run."
    fig = _plot_sweep(payload)
    runs = sorted([d.name for d in _RESULTS_DIR.iterdir() if d.is_dir()])
    return fig, gr.update(choices=runs, value=run_dir.name), ""


with gr.Blocks(title="attention_not — pass_3") as demo:
    gr.Markdown("## attention_not — pass_3: Hand-built NOT via basis projection")

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(label="Run", choices=[], value=None, interactive=True)
            refresh_btn = gr.Button("Refresh runs")
        status = gr.Markdown()
        plot = gr.Plot()

        def _on_load():
            run_dir = _find_latest_run()
            if run_dir is None:
                return None, gr.update(choices=[], value=None), "No runs found."
            payload = _load_payload(run_dir)
            if payload is None:
                return None, gr.update(choices=[], value=None), "No benchmark.json."
            fig = _plot_sweep(payload)
            runs = sorted([d.name for d in _RESULTS_DIR.iterdir() if d.is_dir()])
            return fig, gr.update(choices=runs, value=run_dir.name), f"Loaded {run_dir.name}"

        def _on_select(run_name):
            return _refresh_demo(run_name)

        def _on_refresh():
            runs = sorted([d.name for d in _RESULTS_DIR.iterdir() if d.is_dir()])
            if not runs:
                return gr.update(choices=[], value=None), "No runs found."
            return gr.update(choices=runs, value=runs[-1]), ""

        demo.load(_on_load, inputs=None, outputs=[plot, run_dd, status])
        run_dd.change(_on_select, inputs=[run_dd], outputs=[plot, run_dd, status])
        refresh_btn.click(_on_refresh, inputs=None, outputs=[run_dd, status])

    with gr.Tab("Benchmark"):
        benchmark_panel(_GOAL_DIR)

if __name__ == "__main__":
    demo.launch()