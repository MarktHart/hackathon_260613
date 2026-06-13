"""Gradio app for attention_quantile / pass_3.

Demo tab: scaled-dot-product attention's tail structure across the sweep, the
causal ablations, and a Lorenz curve showing how concentrated the canonical
condition is. Benchmark tab: the cross-attempt leaderboard.
"""

import os
import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
GOAL_DIR = os.path.dirname(HERE)

PARETO_COLOR = "#c0392b"
EXP_COLOR = "#2980b9"


def list_runs():
    if not os.path.isdir(RESULTS):
        return []
    runs = [d for d in os.listdir(RESULTS) if os.path.isdir(os.path.join(RESULTS, d))]
    return sorted(runs, reverse=True)


def load_artefact(run):
    if not run:
        return None
    path = os.path.join(RESULTS, run, "artefact.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _empty_fig(msg):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return fig


def fig_sweep(art):
    if art is None:
        return _empty_fig("no run found — execute main.py first")
    sweep = art["sweep"]
    ids = [r["condition_id"] for r in sweep]
    ratios = [r["quantile_ratio"] for r in sweep]
    colors = [PARETO_COLOR if r["tail_type"] == "pareto" else EXP_COLOR for r in sweep]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(ids)), ratios, color=colors, edgecolor="white")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="uniform baseline (=1.0)")
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels(ids, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("quantile ratio  (q90 / q50)")
    ax.set_title("Attention tail concentration across the sweep")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PARETO_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=EXP_COLOR),
    ]
    ax.legend(handles + [ax.lines[0]], ["pareto (heavy tail)", "exponential (light tail)", "baseline"], fontsize=8)
    fig.tight_layout()
    return fig


def fig_ablation(art):
    if art is None:
        return _empty_fig("no run found — execute main.py first")
    abl = art["ablations"]
    order = ["full", "no_temperature", "linear_no_exp", "uniform_baseline"]
    labels = ["full\n(softmax+temp)", "ablate\ntemperature", "ablate exp\n(linear)", "uniform\nbaseline"]
    lifts = [abl[k]["pareto_vs_exponential_lift"] for k in order]
    canon = [abl[k]["canonical_quantile_ratio"] for k in order]
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x - w / 2, lifts, w, color="#8e44ad", label="pareto/exp lift")
    ax.bar(x + w / 2, canon, w, color="#16a085", label="canonical q-ratio")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ratio")
    ax.set_title("Causal ablation: knocking out the circuit collapses the tail")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_lorenz(art):
    if art is None:
        return _empty_fig("no run found — execute main.py first")
    lor = np.array(art["lorenz_canonical"])
    uni = np.array(art["lorenz_uniform"])
    n = len(lor)
    frac = np.arange(1, n + 1) / n
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(frac, lor, color=PARETO_COLOR, linewidth=2, label="canonical attention (pareto_0p5)")
    ax.plot(frac, uni, color="gray", linestyle="--", linewidth=1.5, label="uniform attention")
    ax.fill_between(frac, uni, lor, color=PARETO_COLOR, alpha=0.12)
    ax.set_xlabel("fraction of keys (sorted by attention, descending)")
    ax.set_ylabel("cumulative attention mass")
    ax.set_title("Lorenz curve: a few keys dominate the canonical condition")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def summary_md(art):
    if art is None:
        return "**No run found.** Run `main.py` to produce `results/<ts>/artefact.json`."
    f = art["ablations"]["full"]
    nt = art["ablations"]["no_temperature"]
    ln = art["ablations"]["linear_no_exp"]
    return (
        f"**Mechanism:** `attn = softmax(GAIN·scale·Q·Kᵀ)`, GAIN=`{art['gain']}` "
        f"(single attention head, no MLP).\n\n"
        f"| variant | canonical q-ratio | pareto/exp lift |\n"
        f"|---|---|---|\n"
        f"| **full** | {f['canonical_quantile_ratio']:.2f} | {f['pareto_vs_exponential_lift']:.2f} |\n"
        f"| ablate temperature | {nt['canonical_quantile_ratio']:.2f} | {nt['pareto_vs_exponential_lift']:.2f} |\n"
        f"| ablate exp (linear) | {ln['canonical_quantile_ratio']:.2f} | {ln['pareto_vs_exponential_lift']:.2f} |\n"
        f"| uniform baseline | 1.00 | 1.00 |\n\n"
        f"Freezing the per-condition temperature drives the pareto/exp lift back to ~1.0 — "
        f"the temperature scale is causally responsible for the heavy tail."
    )


def refresh(run):
    art = load_artefact(run)
    return fig_sweep(art), fig_ablation(art), fig_lorenz(art), summary_md(art)


_runs = list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_quantile · pass_3\n"
        "**Hand-built scaled-dot-product attention.** The heavy/light tail of the "
        "attention distribution is governed by the per-condition temperature `scale`; "
        "the quantile ratio (q90/q50) tracks it. Ablations show the mechanism is the cause."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            run_dd = gr.Dropdown(choices=_runs, value=_default, label="results run")
            md = gr.Markdown(summary_md(load_artefact(_default)))
            with gr.Row():
                p_sweep = gr.Plot(label="Sweep: quantile ratio per condition")
                p_abl = gr.Plot(label="Causal ablation")
            p_lor = gr.Plot(label="Lorenz curve (canonical condition)")

            run_dd.change(refresh, inputs=[run_dd], outputs=[p_sweep, p_abl, p_lor, md])
            demo.load(refresh, inputs=[run_dd], outputs=[p_sweep, p_abl, p_lor, md])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
