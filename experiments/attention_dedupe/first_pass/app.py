"""
Gradio app for the hand-built duplicate-token head.

Demo tab:
  - an attention heatmap for a chosen sequence at the canonical dup_rate, with
    the ground-truth previous-occurrence target marked (▲) so a human can check
    by eye that mass lands on the right key;
  - a bar chart of dedup_mass vs the uniform-causal baseline across the dup-rate
    sweep.

Benchmark tab:
  - the shared cross-attempt leaderboard / history panel.
"""

import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"


def list_runs():
    if not RESULTS.exists():
        return []
    runs = [p.name for p in RESULTS.iterdir()
            if p.is_dir() and (p / "example.npz").exists()]
    return sorted(runs, reverse=True)


def _load(run_id):
    run_dir = RESULTS / run_id
    ex = np.load(run_dir / "example.npz")
    payload = None
    pj = run_dir / "payload.json"
    if pj.exists():
        payload = json.loads(pj.read_text())
    return ex, payload


def heatmap(run_id, seq_idx):
    if not run_id:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "no runs yet", ha="center", va="center")
        ax.axis("off")
        return fig
    ex, _ = _load(run_id)
    tokens, prev, attn = ex["tokens"], ex["prev"], ex["attn"]
    n = tokens.shape[0]
    s = int(max(0, min(seq_idx, n - 1)))
    A = attn[s]
    L = A.shape[0]

    fig, ax = plt.subplots(figsize=(6, 5.2))
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("key position k")
    ax.set_ylabel("query position q")
    ax.set_title(f"seq {s}: attention (▲ = previous occurrence target)")
    # mark ground-truth target for each duplicate query
    for q in range(L):
        p = int(prev[s, q])
        if p >= 0:
            ax.scatter(p, q, marker="^", s=42, edgecolors="cyan",
                       facecolors="none", linewidths=1.4)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attn weight")
    fig.tight_layout()
    return fig


def sweep_chart(run_id):
    fig, ax = plt.subplots(figsize=(6, 4))
    if not run_id:
        ax.text(0.5, 0.5, "no runs yet", ha="center", va="center")
        ax.axis("off")
        return fig
    _, payload = _load(run_id)
    if payload is None:
        ax.text(0.5, 0.5, "no payload.json", ha="center", va="center")
        ax.axis("off")
        return fig
    rates = [r["dup_rate"] for r in payload["sweep"]]
    mass = [r["dedup_mass"] for r in payload["sweep"]]
    base = [r["baseline_dedup_mass"] for r in payload["sweep"]]
    x = np.arange(len(rates))
    w = 0.38
    ax.bar(x - w / 2, mass, w, label="dedupe head", color="#d6456b")
    ax.bar(x + w / 2, base, w, label="uniform-causal baseline", color="#888")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.1f}" for r in rates])
    ax.set_xlabel("duplicate density (dup_rate)")
    ax.set_ylabel("mean attn mass on previous occurrence")
    ax.set_ylim(0, 1.05)
    ax.set_title("dedup_mass vs uniform baseline across the sweep")
    ax.legend()
    fig.tight_layout()
    return fig


def refresh(run_id):
    return heatmap(run_id, 0), sweep_chart(run_id)


with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention Deduplication — hand-built duplicate-token head\n"
        "A single causal attention layer routes each repeated token's query "
        "back to its **previous occurrence**. The ▲ marks the ground-truth "
        "target key; bright cells on the ▲ mean the circuit works."
    )

    with gr.Tab("Demo"):
        runs = list_runs()
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=runs, value=(runs[0] if runs else None),
                label="run", scale=3,
            )
            seq_sl = gr.Slider(0, 63, value=0, step=1, label="sequence index", scale=2)
        hm = gr.Plot(label="attention heatmap")
        sw = gr.Plot(label="sweep: dedup mass vs baseline")

        run_dd.change(refresh, inputs=run_dd, outputs=[hm, sw])
        seq_sl.change(heatmap, inputs=[run_dd, seq_sl], outputs=hm)
        demo.load(refresh, inputs=run_dd, outputs=[hm, sw])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
