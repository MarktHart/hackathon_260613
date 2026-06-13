"""Gradio app for attention_astar / pass_2.

Demo tab: load a saved example grid, overlay the hand-built A* attention, mark
the cell A* would expand next, and show the component ablation (full g+h vs
g-only vs h-only vs uniform) at the canonical density. Benchmark tab: the shared
cross-attempt panel.
"""

import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def _runs():
    if not RESULTS.exists():
        return []
    return sorted([p.name for p in RESULTS.iterdir() if p.is_dir()], reverse=True)


def _load(run):
    d = RESULTS / run
    npz = np.load(d / "demo_example.npz")
    abl = json.loads((d / "ablation_summary.json").read_text())
    return npz, abl


def _grid_fig(run):
    npz, _ = _load(run)
    obst, attn, fvals = npz["obstacle"], npz["attn"], npz["f_values"]
    agent, goal = npz["agent"], npz["goal"]
    opt = npz["optimal_next"].reshape(-1, 2)
    N = obst.shape[0]
    peak = np.unravel_index(int(np.argmax(attn)), attn.shape)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for a, data, title in ((ax[0], attn, "Hand-built A* attention"),
                           (ax[1], np.where(fvals >= 0, fvals, np.nan), "A* f-value (g+h)")):
        im = a.imshow(data, cmap="viridis" if title.startswith("Hand") else "magma_r")
        a.set_title(title)
        a.set_xticks(range(N)); a.set_yticks(range(N))
        for (r, c) in [tuple(agent)]:
            a.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, edgecolor="cyan", lw=2))
        a.add_patch(plt.Rectangle((goal[1] - .5, goal[0] - .5), 1, 1, fill=False, edgecolor="lime", lw=2))
        for (r, c) in opt:
            a.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, edgecolor="white", lw=1.5, ls="--"))
        a.scatter([peak[1]], [peak[0]], marker="*", s=220, c="red", edgecolor="k")
        # mark obstacles
        ys, xs = np.where(obst > 0.5)
        a.scatter(xs, ys, marker="x", c="gray", s=40)
        fig.colorbar(im, ax=a, fraction=0.046)
    fig.suptitle("cyan=agent  green=goal  white-dashed=optimal next  red★=attention argmax  x=obstacle")
    fig.tight_layout()
    return fig


def _abl_fig(run):
    _, abl = _load(run)
    names = list(abl.keys())
    align = [abl[n]["alignment"] for n in names]
    top1 = [abl[n]["top1"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(names, align, color=["#2c7", "#caa", "#cca", "#999"])
    ax[0].set_title("heuristic_alignment (Spearman vs -f)")
    ax[0].axhline(0, color="k", lw=.5)
    ax[1].bar(names, top1, color=["#2c7", "#caa", "#cca", "#999"])
    ax[1].set_title("top1_optimal_rate")
    for a in ax:
        a.tick_params(axis="x", rotation=20)
    fig.suptitle("Canonical density 0.2 — full circuit vs ablations vs uniform baseline")
    fig.tight_layout()
    return fig


def _render(run):
    if not run:
        return None, None, "No runs found — run main.py first."
    npz, abl = _load(run)
    txt = "  |  ".join(f"{k}: align={v['alignment']:.3f}, top1={v['top1']:.2f}"
                       for k, v in abl.items())
    return _grid_fig(run), _abl_fig(run), txt


with gr.Blocks() as demo:
    gr.Markdown("# attention_astar / pass_2 — hand-built A* attention circuit\n"
                "Single attention head whose logits are the negative A* f-value "
                "(`f = g + h`), computed on the GPU. Mass lands on low-f cells; "
                "the argmax (red ★) is the cell A* expands next.")
    with gr.Tab("Demo"):
        runs = _runs()
        dd = gr.Dropdown(runs, value=runs[0] if runs else None, label="Run")
        info = gr.Markdown()
        grid_plot = gr.Plot(label="Attention vs A* f-value")
        abl_plot = gr.Plot(label="Component ablation")
        dd.change(_render, inputs=dd, outputs=[grid_plot, abl_plot, info])
        demo.load(_render, inputs=dd, outputs=[grid_plot, abl_plot, info])
    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
