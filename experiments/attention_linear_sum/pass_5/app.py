"""Demo + Benchmark app for attention_linear_sum / pass_5.

Demo tab tells the whole story in one figure:
  (a) a BAR chart — canonical R² of the linear-attention circuit (ours) vs the
      softmax strawman vs the broadcast-ablated circuit vs the mean baseline.
      The single comparison the goal asks for: does the mechanism beat the
      strawman that keeps softmax, and does knocking out the broadcast head
      break it?
  (b) an OPERATING-RANGE line — R² as |α|=|β| sweeps 0.25 → 32 (>2 orders of
      magnitude). Linear stays flat at 1; softmax decays.
  (c)+(d) two R² heatmaps over the 24-pair (α,β) grid: linear (green
      everywhere) vs softmax (red wherever |α|+|β|≠1 or a coeff is negative).
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


def _load(run_id):
    return json.loads((RESULTS / run_id / "viz.json").read_text())


def _grid(sweep):
    g = np.full((len(VALS), len(VALS)), np.nan)
    for rec in sweep.values():
        g[VALS.index(rec["alpha"]), VALS.index(rec["beta"])] = rec["r2"]
    return g


def _heatmap(ax, grid, title):
    im = ax.imshow(grid, vmin=-1, vmax=1, cmap="RdYlGn")
    ax.set_xticks(range(len(VALS))); ax.set_xticklabels([f"{v:g}" for v in VALS])
    ax.set_yticks(range(len(VALS))); ax.set_yticklabels([f"{v:g}" for v in VALS])
    ax.set_xlabel("β"); ax.set_ylabel("α"); ax.set_title(title)
    for i in range(len(VALS)):
        for j in range(len(VALS)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
    return im


def render(run_id):
    if not run_id:
        fig = plt.figure(figsize=(5, 2))
        fig.text(0.5, 0.5, "No runs yet — run main.py", ha="center")
        return fig
    d = _load(run_id)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))

    # (a) bar chart
    ax = axes[0, 0]
    items = list(d["canonical_r2"].items())
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#2e7d32", "#c62828", "#ef6c00", "#9e9e9e"][:len(names)]
    bars = ax.bar(range(len(names)), vals, color=colors)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("R²  (canonical α=β=1)")
    ax.set_title("Mechanism vs strawman vs ablation")
    ax.set_ylim(min(-0.2, min(vals) - 0.1), 1.1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02 if v >= 0 else v - 0.08,
                f"{v:.2f}", ha="center", fontsize=8)

    # (b) operating range
    ax = axes[0, 1]
    for key, style in [("linear", ("#2e7d32", "o", "linear (ours)")),
                       ("softmax", ("#c62828", "s", "softmax strawman"))]:
        pts = d["op_range"][key]
        ax.plot([p["scale"] for p in pts], [p["r2"] for p in pts],
                style[1] + "-", color=style[0], label=style[2])
    ax.set_xscale("log")
    ax.set_xlabel("coefficient magnitude |α|=|β|  (log)")
    ax.set_ylabel("R²")
    ax.set_ylim(-1.05, 1.1)
    ax.axhline(1.0, color="k", lw=0.5, ls=":")
    ax.set_title("Operating range (>2 orders of magnitude)")
    ax.legend(fontsize=8, loc="lower left")

    # (c)+(d) heatmaps
    im = _heatmap(axes[1, 0], _grid(d["sweep_linear"]),
                  "Linear-attention head  R²  (ours)")
    _heatmap(axes[1, 1], _grid(d["sweep_softmax"]), "Softmax head  R²  (strawman)")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, label="R²")

    fig.suptitle("attention_linear_sum · pass_5 — softmax-free summation head", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_linear_sum · pass_5\n"
        "A **2-layer attention-only** circuit (no MLP, no norm). **Layer 1** is a "
        "standard softmax *copy* head that broadcasts the coefficient token "
        "(α,β) from position 2 to every target — the broadcast is computed BY "
        "attention, not hand-placed. **Layer 2** is the same head with **softmax "
        "removed** (linear attention): query=(α,β), one-hot position keys, value "
        "carries the feature, so Σ score·v = α·x₁+β·x₂ exactly. The only delta "
        "from `base_model.py` is dropping softmax on the summation head."
    )
    with gr.Tab("Demo"):
        runs = list_runs()
        run_dd = gr.Dropdown(choices=runs, value=(runs[0] if runs else None),
                             label="Run")
        plot = gr.Plot()
        run_dd.change(render, inputs=run_dd, outputs=plot)
        demo.load(render, inputs=run_dd, outputs=plot)
    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
