"""Gradio app for pass_2 — multiplicative attention head.

Demo tab: for the selected run, three views answer the goal's question.
  1. Per-K routing accuracy: trained multiplicative head vs the SAME head with
     ⊙ swapped for + (ablation) vs the task's additive baseline vs hand-built
     circuit. The ablation collapsing is the causal/faithfulness evidence.
  2. Canonical-K scatter: predicted attended integer value vs true product a*b.
     Points on the diagonal = the head routed to a*b.
  3. Training curve: held-out routing accuracy over steps.
Benchmark tab: shared leaderboard across all attempts at this goal.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np

from agentic.experiments import benchmark_panel, results_dir

GOAL_DIR = Path(__file__).parent.parent
RES = results_dir(__file__).parent if (results_dir(__file__).parent.name == "results") else None


def _runs():
    rd = Path(__file__).parent / "results"
    if not rd.exists():
        return []
    runs = [p for p in rd.iterdir() if p.is_dir() and (p / "extras.json").exists()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def _load(run_name):
    runs = _runs()
    sel = next((r for r in runs if r.name == run_name), runs[0] if runs else None)
    if sel is None:
        return None
    with open(sel / "extras.json") as f:
        return json.load(f)


def _bar(ex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = ex["k_sweep"]
    s = ex["series"]
    x = np.arange(len(ks))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - 1.5 * w, s["trained"], w, label="Trained head (a⊙b, multiplies)", color="#1a9850")
    ax.bar(x - 0.5 * w, s["handbuilt"], w, label="Hand-built bilinear circuit", color="#91cf60")
    ax.bar(x + 0.5 * w, s["baseline"], w, label="Additive baseline (a+b)", color="#fc8d59")
    ax.bar(x + 1.5 * w, s["ablation"], w, label="Ablation: ⊙→+ (same weights)", color="#d73027")
    ax.axhline(1.0 / ex["n_positions"], ls=":", c="gray", lw=1, label="chance (1/16)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("operand range K")
    ax.set_ylabel("routing accuracy  (attend to a·b)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Can the head single out a·b?  Multiplication vs addition")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _scatter(ex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = ex["scatter"]
    tp = np.array(sc["true_product"])
    pv = np.array(sc["pred_value"])
    ok = np.array(sc["correct"])
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    lim = max(tp.max(), pv.max(), 1) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", c="gray", lw=1, label="y = x (routed to a·b)")
    ax.scatter(tp[ok], pv[ok], s=24, c="#1a9850", alpha=0.7, label=f"correct ({ok.sum()})")
    if (~ok).any():
        ax.scatter(tp[~ok], pv[~ok], s=36, c="#d73027", marker="x", label=f"miss ({(~ok).sum()})")
    ax.set_xlabel("true product  a·b")
    ax.set_ylabel("integer value the head attended to")
    ax.set_title(f"Canonical K={ex['canonical_k']}: attended value vs product")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _curve(ex):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = ex["train_curve"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(c["steps"], c["val_acc"], "-o", c="#1a9850", ms=4)
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out routing acc")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Training (rank={ex['rank']}, beta={ex['beta_trained']:.1f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _render(run_name):
    ex = _load(run_name)
    if ex is None:
        return None, None, None, "No runs yet — run main.py first."
    tr = float(np.mean(ex["series"]["trained"]))
    ab = float(np.mean(ex["series"]["ablation"]))
    bl = float(np.mean(ex["series"]["baseline"]))
    md = (
        f"**Mean routing accuracy** — trained head **{tr:.3f}**, "
        f"additive ablation (same weights) **{ab:.3f}**, additive baseline {bl:.3f}, "
        f"chance {1/ex['n_positions']:.3f}.  "
        f"Removing the elementwise product drops the head from {tr:.3f} to {ab:.3f} "
        f"— the multiplication is doing the routing."
    )
    return _bar(ex), _scatter(ex), _curve(ex), md


with gr.Blocks(title="attention_int_mul — pass_2") as demo:
    gr.Markdown(
        "# attention_int_mul · pass_2\n"
        "A single attention head whose query is a **multiplicative (Hadamard) "
        "interaction** of the two operand embeddings — the smallest delta from "
        "`base_model.py` that lets one head compute `a·b`. Trained with a routing "
        "cross-entropy; verified by ablating the product back to a sum."
    )
    with gr.Tab("Demo"):
        runs = _runs()
        choices = [r.name for r in runs] if runs else ["(no runs)"]
        run_dd = gr.Dropdown(choices=choices, value=choices[0], label="run", interactive=bool(runs))
        summary = gr.Markdown()
        bar_plot = gr.Plot(label="routing accuracy by K")
        with gr.Row():
            scatter_plot = gr.Plot(label="attended value vs product (canonical K)")
            curve_plot = gr.Plot(label="training curve")
        run_dd.change(_render, inputs=[run_dd], outputs=[bar_plot, scatter_plot, curve_plot, summary])
        demo.load(_render, inputs=[run_dd], outputs=[bar_plot, scatter_plot, curve_plot, summary])
    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()
