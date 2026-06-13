"""Demo + Benchmark for attention_dijkstra / pass_2 (soft-min attention relaxation)."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)
RESULTS_DIR = os.path.join(ATTEMPT_DIR, "results")


# --------------------------------------------------------------------------- #
# Run discovery / loading                                                     #
# --------------------------------------------------------------------------- #
def list_runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = [d for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return sorted(runs, reverse=True)


def _load_json(run, name):
    p = os.path.join(RESULTS_DIR, run, name)
    if not os.path.isfile(p):
        return None
    with open(p) as f:
        return json.load(f)


def _load_npz(run, name):
    p = os.path.join(RESULTS_DIR, run, name)
    if not os.path.isfile(p):
        return None
    return np.load(p, allow_pickle=False)


def _empty_fig(msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return fig


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def plot_hop_ablation(run):
    """Causal / faithfulness: accuracy vs number of relaxation hops, per size."""
    abl = _load_json(run, "hop_ablation.json")
    if abl is None:
        return _empty_fig("no hop_ablation.json in this run")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in sorted(abl["curves"], key=lambda x: int(x)):
        hops = abl["hops"][n]
        accs = abl["curves"][n]
        ax.plot(hops, accs, marker=".", label=f"n={n}")
    base = abl.get("onehop_baseline_canonical")
    if base is not None:
        ax.axhline(base, color="grey", ls="--", lw=1,
                   label=f"one-hop baseline (n=16) = {base:.2f}")
    ax.set_xlabel("number of soft-min relaxation hops")
    ax.set_ylabel("distance accuracy")
    ax.set_title("Propagation depth is causal: knock out hops → accuracy collapses")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pred_scatter(run):
    """Correctness + strawman: predicted vs true Dijkstra distance, one graph."""
    ex = _load_npz(run, "example_graph.npz")
    if ex is None:
        return _empty_fig("no example_graph.npz in this run")
    mask = ex["mask"].astype(bool)
    true = ex["true"][mask]
    pred = ex["pred"][mask]
    onehop = ex["onehop"][mask]
    onehop = np.where(np.isfinite(onehop), onehop, np.nan)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hi = float(np.nanmax([true.max(), np.nanmax(pred)])) * 1.05 + 1e-6
    ax.plot([0, hi], [0, hi], color="black", lw=1, ls="--", label="y = x (exact)")
    ax.scatter(true, onehop, color="tab:orange", alpha=0.7, s=40,
               label="one-hop baseline")
    ax.scatter(true, pred, color="tab:blue", alpha=0.85, s=40,
               label="soft-min attention (n-1 hops)")
    ax.set_xlabel("true Dijkstra distance")
    ax.set_ylabel("predicted distance")
    ax.set_title(f"Recovers exact distances (n={int(ex['n'])} graph)")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_beta_sweep(run):
    """Temperature: soft (low beta) underestimates → hard (high beta) is exact."""
    bsw = _load_json(run, "beta_sweep.json")
    if bsw is None:
        return _empty_fig("no beta_sweep.json in this run")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(bsw["betas"], bsw["accuracy"], marker="o", color="tab:blue",
            label="distance accuracy")
    ax.plot(bsw["betas"], bsw["order_corr"], marker="s", color="tab:green",
            label="order correlation")
    ax.set_xscale("log")
    ax.set_xlabel("attention temperature  β   (β→∞ = hard Dijkstra)")
    ax.set_ylabel("metric @ canonical n=16")
    ax.set_title("It really is a soft-min: low β (soft) blurs, high β (sharp) is exact")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def plot_attention(run):
    """Mechanism: final-hop attention = predecessor (shortest-path) tree."""
    ex = _load_npz(run, "example_graph.npz")
    if ex is None:
        return _empty_fig("no example_graph.npz in this run")
    attn = ex["attn"]
    n = int(ex["n"])
    source = int(ex["source"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(attn, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_xlabel("target node v")
    ax.set_ylabel("attended predecessor u")
    ax.set_title(f"Converged attention softmax(−β·cost)\n(source = node {source})")
    ax.axvline(source, color="cyan", lw=1, ls=":")
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    fig.tight_layout()
    return fig


def render(run):
    if not run:
        e = _empty_fig("no runs yet — execute main.py")
        return e, e, e, e
    return (plot_hop_ablation(run), plot_pred_scatter(run),
            plot_beta_sweep(run), plot_attention(run))


# --------------------------------------------------------------------------- #
# App                                                                         #
# --------------------------------------------------------------------------- #
with gr.Blocks() as demo:
    gr.Markdown("# attention_dijkstra · pass_2")
    gr.Markdown(
        "**Soft-min *attention* relaxation.** A single weight-tied attention "
        "block whose logits are `−β·(dᵤ + wᵤᵥ)` and whose read-out is the "
        "soft-min `−1/β·logsumexp`, applied recurrently for `n−1` hops. "
        "β is the attention temperature (β→∞ ⇒ hard Dijkstra)."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            runs = list_runs()
            run_dd = gr.Dropdown(
                choices=runs, value=(runs[0] if runs else None),
                label="run", interactive=True,
            )
            with gr.Row():
                hop_plot = gr.Plot(label="Faithfulness: accuracy vs #hops")
                scatter_plot = gr.Plot(label="Correctness: pred vs true")
            with gr.Row():
                beta_plot = gr.Plot(label="Temperature: β sweep")
                attn_plot = gr.Plot(label="Mechanism: attention tree")

            run_dd.change(render, inputs=run_dd,
                          outputs=[hop_plot, scatter_plot, beta_plot, attn_plot])
            demo.load(render, inputs=run_dd,
                      outputs=[hop_plot, scatter_plot, beta_plot, attn_plot])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
