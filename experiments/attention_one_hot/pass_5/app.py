"""Gradio app for attention_one_hot / pass_5.

Demo tab makes two claims legible at a glance:
  (A) Measured baseline comparison — the method works (~1.0 one-hot at L=64)
      while the no-temperature, uniform, and causally query-patched strawmen
      all collapse toward 1/L.
  (B) Operating range — under realistic non-orthogonal noise, how the method
      vs ablated variants hold up as the query-key alignment degrades.
"""

import json
from pathlib import Path
from typing import List

import numpy as np
import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

RESULTS_ROOT = Path(__file__).parent / "results"

COLORS = {
    "method": "#d62728",
    "no_temperature": "#1f77b4",
    "corrupted_query": "#9467bd",
    "uniform": "#7f7f7f",
    "linear_no_exp": "#2ca02c",
}
LABELS = {
    "method": "method: softmax(QK/τ)",
    "no_temperature": "ablate τ: softmax(QK)",
    "corrupted_query": "patch query (causal)",
    "uniform": "no-attention (1/L)",
    "linear_no_exp": "ablate exp: relu-norm",
}


def list_runs() -> List[str]:
    if not RESULTS_ROOT.exists():
        return []
    runs = [d for d in RESULTS_ROOT.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def _load(run_name: str, fname: str) -> dict:
    if not run_name:
        return {}
    p = RESULTS_ROOT / run_name / fname
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def make_plot(run_name: str):
    abl = _load(run_name, "ablations.json")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    if not abl:
        for ax in axes:
            ax.text(0.5, 0.5, "No ablations.json found", ha="center", va="center",
                    transform=ax.transAxes)
        return fig

    fig.suptitle(f"One-hot attention: measured baselines & operating range — {run_name}",
                 fontsize=13)

    # ---- Panel A: bar chart, target attention at canonical L=64 ----
    axA = axes[0]
    var = abl["variant_target_attention"]
    order = ["method", "no_temperature", "corrupted_query", "uniform"]
    order = [o for o in order if o in var]
    vals = [var[o].get("64", var[o].get(64)) for o in order]
    xs = np.arange(len(order))
    bars = axA.bar(xs, vals, color=[COLORS[o] for o in order])
    axA.axhline(1.0 / 64, ls="--", color="black", lw=1,
                label="uniform 1/L = 0.0156")
    axA.set_xticks(xs)
    axA.set_xticklabels([LABELS[o] for o in order], rotation=20, ha="right", fontsize=8)
    axA.set_ylabel("target attention @ L=64")
    axA.set_ylim(0, 1.05)
    axA.set_title("(A) Method works; measured strawmen fail")
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontsize=8)
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(True, axis="y", alpha=0.3)

    # ---- Panel B: alignment robustness sweep ----
    axB = axes[1]
    al = abl["alignment_sweep"]
    alphas = al["alphas"]
    for m in ["method", "no_temperature", "linear_no_exp", "uniform"]:
        if m in al:
            axB.plot(alphas, al[m], "o-", color=COLORS[m], label=LABELS[m], ms=4)
    axB.set_xlabel("query · target-key alignment  α  (1 = exact match)")
    axB.set_ylabel("target attention")
    axB.set_ylim(0, 1.05)
    axB.invert_xaxis()
    axB.set_title("(B) Operating range under realistic noise")
    axB.legend(fontsize=8, loc="upper right")
    axB.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention One-Hot — pass_5 (hand-built + measured ablations)\n\n"
        "A **hand-built scaled dot-product head** `softmax((keys · query) / τ)` solves the "
        "one-hot lookup exactly. This attempt adds the two things the prior pass "
        "lacked:\n\n"
        "- **(A) Measured baselines** — the no-temperature softmax, the uniform "
        "no-attention head, and a **causally query-patched** head are all scored "
        "through the *same* evaluator. They collapse to ≈`1/L` while the method "
        "stays ≈1.0.\n"
        "- **(B) Causal faithfulness** — patching (corrupting) the query "
        "activation destroys one-hot, proving the `q·k` dot product is the "
        "load-bearing wire. Panel B sweeps the query-key alignment under "
        "realistic noise to show each ablation's breaking point."
    )

    with gr.Row():
        run_dropdown = gr.Dropdown(
            choices=list_runs(),
            value=list_runs()[0] if list_runs() else None,
            label="Run",
            interactive=True,
            scale=1,
        )
    plot_output = gr.Plot(label="Baselines & operating range")

    run_dropdown.change(fn=make_plot, inputs=run_dropdown, outputs=plot_output)
    demo.load(fn=make_plot, inputs=run_dropdown, outputs=plot_output)

    gr.Markdown("---\n## Benchmark history")
    benchmark_panel("attention_one_hot")


if __name__ == "__main__":
    demo.launch()
