"""Demo + Benchmark Gradio app for attention_linear_sum / pass_4.

Demo tab: two R² heatmaps over the 24 (α, β) sweep — the linear-attention head
(this attempt) vs the softmax strawman — plus a pred-vs-target scatter on the
canonical α=β=1 condition. The contrast is the claim: linear attention is exact
everywhere; softmax collapses wherever |α|+|β| ≠ 1 or a coefficient is negative.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

VALS = [0.0, 1.0, -1.0, 2.0, -2.0]


def list_runs():
    if not RESULTS.is_dir():
        return []
    return sorted((p.name for p in RESULTS.iterdir()
                   if (p / "viz.json").is_file()), reverse=True)


def _load(run_id: str):
    return json.loads((RESULTS / run_id / "viz.json").read_text())


def _grid(sweep: dict) -> np.ndarray:
    """Turn the {"a,b": {...}} dict into a 5x5 R² grid (rows=α, cols=β)."""
    g = np.full((len(VALS), len(VALS)), np.nan)
    for rec in sweep.values():
        i = VALS.index(rec["alpha"])
        j = VALS.index(rec["beta"])
        g[i, j] = rec["r2"]
    return g


def _heatmap(ax, grid: np.ndarray, title: str):
    im = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(VALS))); ax.set_xticklabels([f"{v:g}" for v in VALS])
    ax.set_yticks(range(len(VALS))); ax.set_yticklabels([f"{v:g}" for v in VALS])
    ax.set_xlabel("β"); ax.set_ylabel("α")
    ax.set_title(title)
    for i in range(len(VALS)):
        for j in range(len(VALS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    return im


def render(run_id: str):
    if not run_id:
        fig = plt.figure(figsize=(4, 2))
        fig.text(0.5, 0.5, "No runs yet — run main.py", ha="center")
        return fig
    d = _load(run_id)
    gl = _grid(d["sweep_linear"])
    gs = _grid(d["sweep_softmax"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    im = _heatmap(axes[0], gl, "Linear-attention head  R²  (this attempt)")
    _heatmap(axes[1], gs, "Softmax strawman  R²")
    fig.colorbar(im, ax=axes[1], fraction=0.046, label="R²")

    pred = np.array(d["canonical_pred"])[:, 0]
    tgt = np.array(d["canonical_target"])[:, 0]
    ax = axes[2]
    ax.scatter(tgt, pred, s=10, alpha=0.5)
    lim = [min(tgt.min(), pred.min()), max(tgt.max(), pred.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("target  α·x₁+β·x₂"); ax.set_ylabel("head output")
    r2 = 1.0 - np.mean((pred - tgt) ** 2) / np.var(tgt)
    ax.set_title(f"Canonical α=β=1   R²={r2:.4f}")
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_linear_sum · pass_4 — linear-attention head\n"
        "A single hand-set attention head computes **y = α·x₁ + β·x₂** at every "
        "target position. Coefficients enter the **query** (α,β); position-identity "
        "**keys** select features x₁,x₂ in the **value**; dropping softmax "
        "(linear attention) makes the weighted sum *exact* for any (α,β). "
        "The softmax variant of the same head is shown as the strawman."
    )
    with gr.Tab("Demo"):
        run_dd = gr.Dropdown(choices=list_runs(), value=(list_runs()[0] if list_runs() else None),
                             label="Run")
        plot = gr.Plot()
        run_dd.change(render, inputs=run_dd, outputs=plot)
        demo.load(render, inputs=run_dd, outputs=plot)
    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()