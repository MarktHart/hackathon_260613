"""Gradio app for attention_distance_compare / pass_3.

Demo tab:
  (1) Distance-decay curve — model vs uniform baseline vs bias-ablated control,
      the headline claim + strawman + causal check in one chart.
  (2) Per-(layer, head) decay-slope heatmap — shows the local-vs-global head
      structure the goal asks about.
Benchmark tab: the shared cross-attempt leaderboard.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")


def _list_runs():
    if not os.path.isdir(RESULTS):
        return []
    runs = [r for r in os.listdir(RESULTS)
            if os.path.isfile(os.path.join(RESULTS, r, "demo.json"))]
    return sorted(runs, reverse=True)


def _load(run):
    if not run:
        return None
    p = os.path.join(RESULTS, run, "demo.json")
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _empty_fig(msg):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


def decay_fig(run):
    d = _load(run)
    if d is None:
        return _empty_fig("No run found — execute main.py first.")
    bins = np.array(d["distance_bins"])
    model = np.array(d["mean_attn_per_bin"])
    uni = np.array(d["uniform_baseline_per_bin"])
    abl = np.array(d["ablated_mean_attn_per_bin"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(bins, model, "o-", lw=2.2, color="#1f77b4",
            label=f"model (slope={d['headline_slope']:.2f})")
    ax.plot(bins, abl, "s--", lw=1.8, color="#d62728",
            label=f"bias ablated (slope={d['ablated_slope']:.2f})")
    ax.plot(bins, uni, "^:", lw=1.6, color="#7f7f7f", label="uniform baseline")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("token distance |i - j|  (bin center, log2)")
    ax.set_ylabel("mean attention weight (log)")
    ax.set_title("Distance decay: zeroing the relative-position bias\n"
                 "collapses the model onto the uniform baseline")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def heatmap_fig(run):
    d = _load(run)
    if d is None:
        return _empty_fig("No run found — execute main.py first.")
    slopes = np.array(d["layer_head_slope"])  # (4, 8)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    im = ax.imshow(slopes, cmap="magma", aspect="auto")
    ax.set_xlabel("head  (0 = local  →  7 = global)")
    ax.set_ylabel("layer")
    ax.set_xticks(range(slopes.shape[1]))
    ax.set_yticks(range(slopes.shape[0]))
    ax.set_title("Per-head distance-decay slope\n(bigger = more local)")
    for l in range(slopes.shape[0]):
        for h in range(slopes.shape[1]):
            ax.text(h, l, f"{slopes[l, h]:.1f}", ha="center", va="center",
                    color="white" if slopes[l, h] < slopes.max() * 0.6 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, label="decay slope")
    fig.tight_layout()
    return fig


def refresh(run):
    return decay_fig(run), heatmap_fig(run)


with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention Distance Compare — pass_3\n"
        "Hand-built **relative-position-bias** attention heads. Distance decay "
        "is emitted by a real softmax over QK+bias logits; an **ablation** of the "
        "bias term restores the uniform baseline (causal evidence)."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            runs = _list_runs()
            run_dd = gr.Dropdown(
                choices=runs, value=(runs[0] if runs else None),
                label="run", interactive=True,
            )
            with gr.Row():
                decay_plot = gr.Plot(label="distance decay + ablation")
                heat_plot = gr.Plot(label="per-head locality")
            run_dd.change(refresh, inputs=run_dd, outputs=[decay_plot, heat_plot])
            demo.load(refresh, inputs=run_dd, outputs=[decay_plot, heat_plot])
        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
