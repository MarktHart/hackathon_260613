"""Gradio app for the attention-boundary hand-built circuit (pass_3).

Demo tab: per-head boundary sharpness, the attention heatmap of a chosen head,
the alpha operating-range curve, and the faithfulness ablation. Benchmark tab:
the shared cross-attempt leaderboard.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

DELIM_POS = 8
SEG_LEN = 8


# --------------------------------------------------------------------------
def list_runs():
    if not RESULTS.exists():
        return []
    runs = [p.name for p in RESULTS.iterdir() if (p / "meta.json").exists()]
    runs.sort(reverse=True)
    return runs


def load_run(run_name):
    if not run_name:
        return None, None
    d = RESULTS / run_name
    if not (d / "meta.json").exists():
        return None, None
    meta = json.loads((d / "meta.json").read_text())
    attn = np.load(d / "attn.npy") if (d / "attn.npy").exists() else None
    return meta, attn


def _empty(msg="No run found — run main.py first."):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


# --------------------------------------------------------------------------
def fig_per_head(meta):
    if meta is None:
        return _empty()
    ph = meta["per_head"]
    n = len(ph["segA"]["sharpness"])
    abl = meta["ablation"]["ablated_sharpness"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(n)
    w = 0.38
    ax.bar(x - w / 2, ph["segA"]["sharpness"], w, label="segA queries", color="#2c7fb8")
    ax.bar(x + w / 2, ph["segB"]["sharpness"], w, label="segB queries", color="#7fcdbb")
    ax.axhline(abl, color="crimson", ls="--",
               label=f"segment-feature ablated ({abl:.2f})")
    ax.axhline(0.0, color="gray", ls=":", lw=1, label="uniform baseline (0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"head {i}\n(alpha={a})" for i, a in enumerate(meta["head_alphas"])])
    ax.set_ylabel("boundary sharpness\n(within - max(delim, cross, eos))")
    ax.set_title("Per-head boundary sharpness vs ablated / baseline")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return fig


def fig_heatmap(meta, attn, head):
    if attn is None:
        return _empty()
    head = int(head)
    h = attn[:, head].mean(axis=0)  # (L, L), batch-averaged
    fig, ax = plt.subplots(figsize=(5.6, 5))
    im = ax.imshow(h, cmap="magma", vmin=0, vmax=h.max())
    ax.set_title(f"Head {head} attention (batch-averaged)")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    for p, c, lbl in [(DELIM_POS, "cyan", "DELIM"), (h.shape[0] - 1, "lime", "EOS")]:
        ax.axvline(p, color=c, ls="--", lw=1.2, alpha=0.8)
        ax.axhline(p, color=c, ls="--", lw=0.8, alpha=0.4)
    ax.text(DELIM_POS, -1.2, "DELIM", color="black", ha="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_operating_range(meta):
    if meta is None:
        return _empty()
    sw = meta["alpha_sweep"]
    xs = [r["alpha"] for r in sw]
    ys = [r["sharpness"] for r in sw]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(xs, ys, "o-", color="#2c7fb8")
    ax.axhline(0.0, color="crimson", ls="--", label="uniform baseline")
    ax.set_xscale("log")
    ax.set_xlabel("alpha  (segment-feature strength, log scale, 4 orders)")
    ax.set_ylabel("boundary sharpness")
    ax.set_title("Operating range: sharpness vs feature strength")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_ablation(meta):
    if meta is None:
        return _empty()
    a = meta["ablation"]
    labels = ["full circuit", "segment feature\nablated (alpha=0)", "uniform\nbaseline"]
    vals = [a["full_sharpness"], a["ablated_sharpness"], a["baseline_sharpness"]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(labels, vals, color=["#2c7fb8", "crimson", "gray"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("boundary sharpness")
    ax.set_title("Faithfulness: knock out the segment-sign feature")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


def refresh(run_name, head):
    meta, attn = load_run(run_name)
    return (fig_per_head(meta), fig_heatmap(meta, attn, head),
            fig_operating_range(meta), fig_ablation(meta))


def heatmap_only(run_name, head):
    meta, attn = load_run(run_name)
    return fig_heatmap(meta, attn, head)


# --------------------------------------------------------------------------
_runs = list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks(title="attention_boundary - pass_3") as demo:
    gr.Markdown(
        "# Attention boundary detection — hand-built circuit\n"
        "Real dot-product attention `softmax(Q·Kᵀ)` whose 2-D Q/K features are "
        "derived from the tokens: **segment-sign** relative to the *detected* "
        "delimiter, plus a special-token penalty. Each head concentrates "
        "attention within its own segment. The **ablation** zeroes the "
        "segment-sign feature and the boundary behaviour collapses to the "
        "uniform baseline — causal evidence the mechanism is real."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(choices=_runs, value=_default, label="Run")
                head_dd = gr.Dropdown(choices=["0", "1", "2", "3"], value="2",
                                      label="Head (for heatmap)")
            with gr.Row():
                p_head = gr.Plot(label="Per-head sharpness")
                p_heat = gr.Plot(label="Attention heatmap")
            with gr.Row():
                p_range = gr.Plot(label="Operating range")
                p_abl = gr.Plot(label="Ablation")

            run_dd.change(refresh, [run_dd, head_dd],
                          [p_head, p_heat, p_range, p_abl])
            head_dd.change(heatmap_only, [run_dd, head_dd], [p_heat])
            demo.load(refresh, [run_dd, head_dd],
                      [p_head, p_heat, p_range, p_abl])

        with gr.Tab("Benchmark"):
            benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
