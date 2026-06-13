"""Gradio app for attention_group_compose / first_pass.

Demo tab: the fidelity-vs-noise curve (method vs naive matmul) plus a heatmap
panel showing a concrete A, B -> predicted C / true C composition.

Benchmark tab: the shared cross-attempt leaderboard / history panel.
"""
import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).resolve().parent.parent
ATTEMPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ATTEMPT_DIR / "results"


def _list_runs():
    if not RESULTS_DIR.exists():
        return []
    return sorted([p.name for p in RESULTS_DIR.iterdir() if p.is_dir()], reverse=True)


def _load_run(run_id):
    if not run_id:
        return None, None
    run_dir = RESULTS_DIR / run_id
    sweep, demo_ex = None, None
    sp = run_dir / "sweep.json"
    dp = run_dir / "demo_examples.json"
    if sp.exists():
        sweep = json.loads(sp.read_text())
    if dp.exists():
        demo_ex = json.loads(dp.read_text())
    return sweep, demo_ex


def fidelity_curve(run_id):
    sweep, _ = _load_run(run_id)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not sweep:
        ax.set_title("No sweep data")
        return fig
    noise = [r["noise_level"] for r in sweep]
    method_fid = [max(0.0, 1.0 - r["frobenius_error"]) for r in sweep]
    base_fid = [max(0.0, 1.0 - r["linear_baseline_error"]) for r in sweep]
    ax.plot(noise, method_fid, "o-", color="#1565c0", lw=2, label="snap-to-group")
    ax.plot(noise, base_fid, "s--", color="#c62828", lw=2, label="naive matmul (A@B)")
    ax.axvline(20.0, color="gray", ls=":", alpha=0.7)
    ax.text(20.4, 0.05, "canonical σ=20", color="gray", fontsize=8)
    ax.set_xlabel("noise level σ (logit units)")
    ax.set_ylabel("composition fidelity  (1 − Frobenius err)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Group-law composition vs naive matmul across noise")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def heatmaps(run_id, noise_key):
    _, demo_ex = _load_run(run_id)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    if not demo_ex or noise_key not in demo_ex:
        for ax in axes:
            ax.axis("off")
        axes[0].set_title("No demo data")
        return fig
    ex = demo_ex[noise_key]
    panels = [
        ("A (noisy)", np.array(ex["A"])),
        ("B (noisy)", np.array(ex["B"])),
        ("predicted C", np.array(ex["pred"])),
        ("true C", np.array(ex["true"])),
    ]
    im = None
    for ax, (title, M) in zip(axes, panels):
        im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.012, pad=0.02)
    fig.suptitle(f"Example composition at σ={noise_key}  "
                 f"(snap-to-group recovers the clean permutation)", fontsize=11)
    return fig


def refresh(run_id):
    return fidelity_curve(run_id), heatmaps(run_id, "20.0")


with gr.Blocks() as demo:
    gr.Markdown("# attention_group_compose — snap-to-group composer\n"
                "Do noisy permutation-attention matrices compose by the group law "
                "rather than by softmax-relaxed matmul? We project each input onto "
                "its nearest C₆ rotation, compose exactly in the group "
                "(`k_c = (k_a + k_b) mod n`), and compare to naive `A@B`.")

    runs = _list_runs()
    default_run = runs[0] if runs else None

    with gr.Tab("Demo"):
        run_dd = gr.Dropdown(choices=runs, value=default_run, label="run")
        noise_dd = gr.Dropdown(choices=["0.0", "20.0"], value="20.0",
                               label="example noise level σ")
        curve_plot = gr.Plot(label="fidelity vs noise")
        heat_plot = gr.Plot(label="example composition")

        run_dd.change(refresh, inputs=run_dd, outputs=[curve_plot, heat_plot])
        noise_dd.change(heatmaps, inputs=[run_dd, noise_dd], outputs=heat_plot)
        demo.load(refresh, inputs=run_dd, outputs=[curve_plot, heat_plot])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
