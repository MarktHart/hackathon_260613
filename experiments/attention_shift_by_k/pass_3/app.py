"""Gradio app for attention_shift_by_k / pass_3.

Demo tab tells the whole story in two coordinated views:
  (1) Attention heatmap for the head dedicated to a chosen offset k. A clean
      bright band on the k-th sub-diagonal IS the shift-by-k behaviour: query i
      (row) puts its mass on key i-k (column). This is the smallest artefact
      that, if it moved, would change the claim.
  (2) A grouped bar chart per k: the real circuit's best-head mass vs. the
      causal ABLATION (shift matrix removed from W_K) vs. the uniform baseline.
      The ablation collapsing to ~chance is the faithfulness evidence — the
      offset comes from the shift matrix, not from anywhere else.

Benchmark tab drops in the shared cross-attempt leaderboard panel.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
K_SWEEP = (1, 2, 3, 4, 8)


def list_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return runs


def latest_run_name():
    runs = list_runs()
    return runs[0].name if runs else None


def load_summary(run_name):
    if not run_name:
        return None
    p = RESULTS_DIR / run_name / "summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def load_attn(run_name):
    if not run_name:
        return None
    p = RESULTS_DIR / run_name / "attn_heads.npy"
    if not p.exists():
        return None
    return np.load(p)   # (H, L, L)


def make_heatmap(run_name, k):
    summary = load_summary(run_name)
    attn = load_attn(run_name)
    if summary is None or attn is None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "No run found.\nRun main.py first.",
                ha="center", va="center")
        ax.axis("off")
        return fig
    k = int(k)
    head = summary["head_for_k"][str(k)]
    A = attn[head]   # (L, L)
    L = A.shape[0]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=1, aspect="equal")
    # Mark the ideal shift-by-k diagonal.
    ideal_q = np.arange(k, L)
    ideal_k = ideal_q - k
    ax.plot(ideal_k, ideal_q, color="#39ff14", lw=1.0, alpha=0.7,
            label=f"target key i-{k}")
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_title(f"Head {head} attention  (offset k={k})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attn weight")
    fig.tight_layout()
    return fig


def make_bars(run_name):
    summary = load_summary(run_name)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    if summary is None:
        ax.text(0.5, 0.5, "No run found.\nRun main.py first.",
                ha="center", va="center")
        ax.axis("off")
        return fig
    ks = summary["k_values"]
    mech = [summary["mechanism_mass"][str(k)] for k in ks]
    abl = [summary["ablation_mass"][str(k)] for k in ks]
    uni = [summary["uniform_mass"][str(k)] for k in ks]
    x = np.arange(len(ks))
    w = 0.27
    ax.bar(x - w, mech, w, label="real circuit (best head)", color="#2ca02c")
    ax.bar(x, abl, w, label="ablation: shift removed", color="#d62728")
    ax.bar(x + w, uni, w, label="uniform baseline", color="#7f7f7f")
    ax.axhline(summary["uniform_baseline"], ls="--", color="#444", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel("mass on shift-by-k target (i-k)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Circuit vs. ablation vs. chance across offsets")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def make_metrics_md(run_name):
    summary = load_summary(run_name)
    if summary is None:
        return "No run found. Run `main.py` first."
    rows = []
    for k in summary["k_values"]:
        rows.append(
            f"| {k} | {summary['mechanism_mass'][str(k)]:.4f} | "
            f"{summary['mechanism_argmax_acc'][str(k)]:.4f} | "
            f"{summary['ablation_mass'][str(k)]:.4f} | "
            f"{summary['uniform_mass'][str(k)]:.4f} |"
        )
    table = "\n".join(rows)
    return f"""### Run `{run_name}`

Best-head mass on key `i-k` — real QK circuit vs. ablation (shift matrix
removed from `W_K`) vs. uniform baseline = `{summary['uniform_baseline']:.4f}`.

| k | circuit mass | circuit argmax-acc | ablation mass | uniform mass |
|---|---|---|---|---|
{table}

The circuit is ~1.0 across every offset while the ablation collapses to
chance — the shift comes from the `W_K` shift matrix, not from anywhere else.
"""


def update(run_name, k):
    return make_heatmap(run_name, k), make_bars(run_name), make_metrics_md(run_name)


with gr.Blocks() as demo:
    gr.Markdown("# Attention Shift by k — pass_3")
    gr.Markdown(
        "A **hand-built but real** QK circuit: `attn = softmax((X·W_Q)(X·W_K)ᵀ)`. "
        "The shift-by-k band is *computed* from the projections, not painted on. "
        "Each head is dedicated to one offset `k`; the ablation removes the shift "
        "matrix in `W_K` to prove that is where the offset comes from."
    )

    with gr.Row():
        run_dd = gr.Dropdown(
            choices=[d.name for d in list_runs()],
            value=latest_run_name(),
            label="run",
        )
        k_dd = gr.Dropdown(
            choices=[str(k) for k in K_SWEEP],
            value="1",
            label="offset k (heatmap)",
        )
        refresh = gr.Button("refresh", size="sm")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                heat = gr.Plot(label="attention heatmap")
                bars = gr.Plot(label="circuit vs ablation vs chance")
            metrics = gr.Markdown()
        with gr.TabItem("Benchmark"):
            benchmark_panel(str(GOAL_DIR))

    run_dd.change(update, inputs=[run_dd, k_dd], outputs=[heat, bars, metrics])
    k_dd.change(update, inputs=[run_dd, k_dd], outputs=[heat, bars, metrics])
    refresh.click(
        lambda: gr.update(choices=[d.name for d in list_runs()], value=latest_run_name()),
        outputs=run_dd,
    ).then(update, inputs=[run_dd, k_dd], outputs=[heat, bars, metrics])
    demo.load(update, inputs=[run_dd, k_dd], outputs=[heat, bars, metrics])


if __name__ == "__main__":
    demo.launch()
