"""Gradio app for attention_minimax / pass_2 (confidence-gated minimax head)."""

import glob
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


def _runs():
    paths = sorted(glob.glob(os.path.join(HERE, "results", "*", "demo.json")))
    return paths[::-1]  # newest first


def _load(run_path):
    with open(run_path) as f:
        return json.load(f)


def _run_label(p):
    return os.path.basename(os.path.dirname(p))


def _fig_bars(demo, ai):
    """Bar chart: gated vs strawmen at the selected alpha index."""
    rec = demo["canonical"][ai]
    a = rec["alpha"]
    labels = ["A", "B", "C"]
    series = [
        ("gated (ours)", rec["gated"], "#1b9e77"),
        ("softmax (raw)", rec["softmax_raw"], "#d95f02"),
        ("softmax (scaled)", rec["softmax_scaled"], "#7570b3"),
        ("linear", rec["linear"], "#e7298a"),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    x = np.arange(3)
    w = 0.2
    for i, (name, vals, c) in enumerate(series):
        ax.bar(x + (i - 1.5) * w, vals, w, label=name, color=c)
    ax.axhline(1 / 3, ls="--", c="k", lw=1)
    ax.text(2.55, 1 / 3 + 0.01, "minimax 1/3", fontsize=8, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("attention weight"); ax.set_ylim(0, 0.75)
    ax.set_title(f"α={a:.1f}  |  max_score={rec['max_score']:.3f} "
                 f"(τ={demo['tau']})  |  gate β={rec['beta']:.2f}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def _fig_regret(demo):
    """max_weight (= regret + 1/3) vs alpha, all methods."""
    al = demo["alphas"]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(al, [r["gated_mw"] for r in demo["canonical"]], "o-", c="#1b9e77", label="gated (ours)")
    ax.plot(al, [r["softmax_raw_mw"] for r in demo["canonical"]], "s-", c="#d95f02", label="softmax (raw)")
    ax.plot(al, [r["softmax_scaled_mw"] for r in demo["canonical"]], "^-", c="#7570b3", label="softmax (scaled)")
    ax.plot(al, [r["linear_mw"] for r in demo["canonical"]], "v-", c="#e7298a", label="linear")
    ax.axhline(1 / 3, ls="--", c="k", lw=1, label="minimax optimum (1/3)")
    ax.set_xlabel("α (target similarity)"); ax.set_ylabel("max attention weight")
    ax.set_title("Lower = closer to minimax. Target is never in keys → uniform is optimal everywhere.")
    ax.legend(fontsize=8); ax.set_ylim(0.3, 0.75)
    fig.tight_layout()
    return fig


def _fig_causal(demo):
    """Target-injected: gate opens and head concentrates on the real target."""
    ti = demo["target_injected"]
    al = [r["alpha"] for r in ti]
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(al, [r["weight_on_target"] for r in ti], "o-", c="#1b9e77", label="weight on TARGET key")
    ax.plot(al, [r["beta"] / demo["beta_max"] for r in ti], "s--", c="#666",
            label="gate β (normalised)")
    ax.axhline(0.25, ls=":", c="k", lw=1)
    ax.text(0.02, 0.26, "uniform over 4 keys", fontsize=8)
    ax.set_xlabel("α (target similarity)"); ax.set_ylabel("value")
    ax.set_title("Causal check: splice real TARGET into keys → gate opens → head concentrates")
    ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def _render(run_path, ai):
    if not run_path or not os.path.exists(run_path):
        return None, None, None, "No run found. Execute `python main.py` first."
    demo = _load(run_path)
    ai = int(ai)
    rec = demo["canonical"][ai]
    md = (f"**α = {rec['alpha']:.1f}** &nbsp; max score = {rec['max_score']:.3f} "
          f"(threshold τ = {demo['tau']}) &nbsp; gate β = {rec['beta']:.2f}\n\n"
          f"- gated (ours) regret = **{rec['gated_mw'] - 1/3:+.4f}**\n"
          f"- softmax (raw) regret = {rec['softmax_raw_mw'] - 1/3:+.4f}\n"
          f"- linear regret = {rec['linear_mw'] - 1/3:+.4f}\n\n"
          "Gate is closed (β≈0) whenever the best key is only incidentally "
          "similar → the head emits the uniform minimax distribution.")
    return _fig_bars(demo, ai), _fig_regret(demo), _fig_causal(demo), md


with gr.Blocks() as demo:
    gr.Markdown("# attention_minimax — confidence-gated (minimax) attention head")
    with gr.Tab("Demo"):
        gr.Markdown(
            "A standard softmax head leaks mass onto the most spuriously-similar "
            "distractor. Our head gates the inverse-temperature on an **absolute** "
            "match threshold τ: below it the gate closes and the head spreads "
            "uniformly (minimax-optimal max weight = 1/3). Use the slider to scan α."
        )
        runs = _runs()
        run_dd = gr.Dropdown(
            choices=[(_run_label(p), p) for p in runs],
            value=(runs[0] if runs else None),
            label="run",
        )
        alpha_sl = gr.Slider(0, 10, value=0, step=1,
                             label="α index (0 → α=0.0 canonical, 10 → α=1.0)")
        info = gr.Markdown()
        with gr.Row():
            bars = gr.Plot(label="attention weights at α")
            regret = gr.Plot(label="max weight vs α (all methods)")
        causal = gr.Plot(label="causal check: target injected")

        run_dd.change(_render, [run_dd, alpha_sl], [bars, regret, causal, info])
        alpha_sl.change(_render, [run_dd, alpha_sl], [bars, regret, causal, info])
        demo.load(_render, [run_dd, alpha_sl], [bars, regret, causal, info])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
