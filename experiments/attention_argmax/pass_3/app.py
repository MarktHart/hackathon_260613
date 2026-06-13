"""Gradio app for attention_argmax / pass_3.

Demo tab: interactively shows that the softmax head behaves as a soft argmax
(spikes on the winner) and that the no-`exp` linear head does not, plus a
noise-sweep curve showing where the argmax behaviour breaks.
Benchmark tab: cross-attempt leaderboard / metric history.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel, load_task

# App callbacks may run interactively; prefer GPU but stay importable anywhere.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOAL_DIR = os.path.dirname(_HERE)

task = load_task(__file__)
_D = task._D
_N = task._N
TAU_DEFAULT = 0.25


# --------------------------------------------------------------------------- #
# Heads (GPU)
# --------------------------------------------------------------------------- #
def _softmax_head(q, K, tau):
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    return torch.softmax((Kt @ qt) / tau, dim=0).detach().cpu().numpy()


def _linear_head(q, K):
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    w = torch.relu(Kt @ qt)
    if float(w.sum()) <= 1e-12:
        w = torch.ones_like(w)
    return (w / w.sum()).detach().cpu().numpy()


def _demo_batch(separation, noise, seed):
    rng = np.random.default_rng(int(seed))
    q = rng.normal(size=_D).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    K = (noise * rng.normal(size=(_N, _D))).astype(np.float32)
    widx = int(rng.integers(0, _N))
    K[widx] = K[widx] - np.dot(K[widx], q) * q + separation * q
    return q, K, widx


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _bar_plot(separation, noise, tau, seed):
    q, K, widx = _demo_batch(separation, noise, int(seed))
    sm = _softmax_head(q, K, tau)
    lin = _linear_head(q, K)
    x = np.arange(_N)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, w, title in ((axes[0], sm, f"softmax head (exp), tau={tau:.2f}"),
                         (axes[1], lin, "linear head (NO exp)")):
        colors = ["#cccccc"] * _N
        colors[widx] = "#d62728"
        ax.bar(x, w, color=colors)
        ax.axhline(1.0 / _N, ls="--", lw=1, color="#1f77b4", label="uniform 1/N")
        ax.set_title(f"{title}\nwinner mass = {w[widx]:.3f}")
        ax.set_xlabel("key position")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("attention prob.")
    fig.suptitle(f"Red = true winner (sep={separation:.1f}, noise={noise:.2f})",
                 fontsize=11)
    fig.tight_layout()
    return fig


def _latest_comparison():
    files = sorted(glob.glob(os.path.join(_HERE, "results", "*", "comparison.json")))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _sweep_plot():
    comp = _latest_comparison()
    fig, ax = plt.subplots(figsize=(7, 4))
    if comp is None:
        ax.text(0.5, 0.5, "Run main.py to generate the noise sweep.",
                ha="center", va="center")
        ax.axis("off")
        return fig
    rows = comp["rows"]
    noise = [max(r["noise"], 1e-2) for r in rows]   # for log axis
    ax.plot(noise, [r["softmax_winner_mass"] for r in rows],
            "o-", color="#d62728", label="softmax head (exp)")
    ax.plot(noise, [r["linear_winner_mass"] for r in rows],
            "s-", color="#2ca02c", label="linear head (no exp)")
    ax.axhline(1.0 / comp["N"], ls="--", color="#1f77b4", label="uniform 1/N")
    ax.set_xscale("log")
    ax.set_xlabel("key noise std (log scale, >2 orders of magnitude)")
    ax.set_ylabel(f"winner mass @ sep={comp['separation']}")
    ax.set_title("Argmax fidelity vs noise: exp keeps the winner, no-exp collapses")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_argmax — softmax head is a soft argmax\n"
        "A softmax attention head places its mass on the single highest-similarity "
        "key. The temperature `tau` sets the sharpness: `tau→0` ⇒ one-hot (argmax). "
        "The **left** bars are the real head (`exp`); the **right** bars are a "
        "no-`exp` strawman that fails to concentrate. Red bar = the true winner."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            sep = gr.Slider(0.0, 4.0, value=2.0, step=0.5, label="separation (winner sim)")
            noise = gr.Slider(0.0, 10.0, value=1.0, step=0.1, label="key noise std")
            tau = gr.Slider(0.05, 2.0, value=TAU_DEFAULT, step=0.05, label="temperature tau")
            seed = gr.Slider(0, 50, value=0, step=1, label="seed")
        bar = gr.Plot(label="Per-batch attention distribution")
        gr.Markdown(
            "### Operating range — where argmax behaviour holds\n"
            "Winner mass at fixed separation as key noise grows over >2 orders of "
            "magnitude (averaged over 100 seeds, from `main.py`)."
        )
        sweep = gr.Plot(label="Noise sweep")

        inputs = [sep, noise, tau, seed]
        for c in inputs:
            c.change(_bar_plot, inputs=inputs, outputs=bar)
        demo.load(_bar_plot, inputs=inputs, outputs=bar)
        demo.load(_sweep_plot, inputs=None, outputs=sweep)

    with gr.Tab("Benchmark"):
        benchmark_panel(_GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
