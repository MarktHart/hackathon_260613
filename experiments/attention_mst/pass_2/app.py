"""Gradio app for attention_mst / pass_2.

Demo tab: the headline claim — MST edge-recovery F1 vs noise for the trained
attention denoiser, the no-mechanism baseline (Kruskal on the noisy weights),
and the attention-ablated net. The gap (method above baseline) is denoising;
the ablated curve collapsing toward the baseline is the faithfulness evidence
that the net actually uses its attention block. Benchmark tab: cross-attempt
leaderboard.
"""

import os
import json
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

ATT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ATT_DIR, "results")
GOAL_DIR = os.path.dirname(ATT_DIR)


def list_runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = []
    for d in sorted(os.listdir(RESULTS_DIR), reverse=True):
        if os.path.isfile(os.path.join(RESULTS_DIR, d, "summary.json")):
            runs.append(d)
    return runs


def load_summary(run):
    with open(os.path.join(RESULTS_DIR, run, "summary.json")) as f:
        return json.load(f)


def plot_run(run):
    if not run:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "No run yet — execute main.py", ha="center", va="center")
        ax.axis("off")
        return fig

    s = load_summary(run)
    nl = s["noise_levels"]
    can = s.get("canonical_noise", 0.5)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    def panel(ax, key_m, key_b, key_a, title, ylab, ylim=None, hline=None):
        ax.plot(nl, s[key_m], "o-", color="C0", lw=2, label="trained denoiser")
        if key_a in s:
            ax.plot(nl, s[key_a], "^:", color="C2", lw=2, label="attention ablated")
        ax.plot(nl, s[key_b], "s--", color="C3", lw=2, label="baseline (noisy Kruskal)")
        ax.axvline(can, color="gray", ls=":", alpha=0.6)
        if hline is not None:
            ax.axhline(hline, color="green", ls=":", alpha=0.5)
        ax.set_xlabel("noise level")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    panel(axes[0], "method_f1", "baseline_f1", "ablate_f1",
          "MST edge-recovery F1 (headline)", "edge F1", ylim=(0, 1.05))
    panel(axes[1], "method_auroc", "baseline_auroc", "ablate_auroc",
          "edge-ranking AUROC", "AUROC", ylim=(0.4, 1.05))
    panel(axes[2], "method_wratio", "baseline_wratio", "ablate_wratio",
          "MST weight ratio (lower better)", "pred/true weight", hline=1.0)

    fig.suptitle(f"run {run}", fontsize=11)
    fig.tight_layout()
    return fig


def summary_text(run):
    if not run:
        return "No run available."
    s = load_summary(run)
    nl = s["noise_levels"]
    ci = nl.index(s.get("canonical_noise", 0.5))
    mean_m = sum(s["method_f1"]) / len(s["method_f1"])
    mean_b = sum(s["baseline_f1"]) / len(s["baseline_f1"])
    lift = s["method_f1"][ci] - s["baseline_f1"][ci]
    return (f"**Canonical noise {nl[ci]:.1f}** — method F1 {s['method_f1'][ci]:.3f} "
            f"vs baseline {s['baseline_f1'][ci]:.3f}  (lift {lift:+.3f})\n\n"
            f"**mst_recovery (mean F1)** — method {mean_m:.3f} vs baseline {mean_b:.3f}")


_runs = list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks() as demo:
    gr.Markdown("# attention_mst / pass_2 — trained attention denoiser")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown(
                "Does the mechanism **denoise** — recover the planted MST better "
                "than running Kruskal on the noisy weights? The blue curve above "
                "the red baseline is the claim; the green *ablated* curve "
                "collapsing toward red shows the net relies on its attention block."
            )
            run_dd = gr.Dropdown(choices=_runs, value=_default, label="run")
            plot = gr.Plot(label="F1 / AUROC / weight-ratio vs noise")
            txt = gr.Markdown()

            run_dd.change(plot_run, inputs=run_dd, outputs=plot)
            run_dd.change(summary_text, inputs=run_dd, outputs=txt)
            demo.load(plot_run, inputs=run_dd, outputs=plot)
            demo.load(summary_text, inputs=run_dd, outputs=txt)

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt leaderboard")
            try:
                benchmark_panel(GOAL_DIR)
            except Exception as e:  # never break the boot-check
                gr.Markdown(f"benchmark panel unavailable: {e}")

if __name__ == "__main__":
    demo.launch()
