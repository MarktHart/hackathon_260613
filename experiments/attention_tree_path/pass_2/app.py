"""Gradio app for attention_tree_path / pass_2.

Demo tab: interactive attention heatmap of the hand-built QK content-addressing
circuit, a per-query attention bar with the ground-truth target marked, a causal
ablation bar, and the depth/rule operating-range sweep.
Benchmark tab: shared cross-attempt leaderboard.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"


# ──────────────────────────────────────────────────────────────────────
# data loading
# ──────────────────────────────────────────────────────────────────────
def get_runs():
    if not RESULTS.exists():
        return []
    return sorted(
        [d.name for d in RESULTS.iterdir() if (d / "meta.json").exists()],
        reverse=True,
    )


def load_meta(run):
    if not run:
        return None
    p = RESULTS / run / "meta.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cond_keys(meta):
    return [c["key"] for c in meta["conditions"]] if meta else []


def get_cond(meta, key):
    for c in meta["conditions"]:
        if c["key"] == key:
            return c
    return meta["conditions"][0]


def load_attn(run, key):
    p = RESULTS / run / f"attn_{key}.npy"
    return np.load(p) if p.exists() else None


def load_benchmark(run):
    p = RESULTS / run / "benchmark.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ──────────────────────────────────────────────────────────────────────
# figures
# ──────────────────────────────────────────────────────────────────────
def fig_heatmap(attn, cond, q):
    L = attn.shape[0]
    q = min(max(int(q), 0), L - 1)
    tp = cond["target_pos"]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    ax.set_xlabel("Key position (node id)")
    ax.set_ylabel("Query position (node id)")
    step = 1 if L <= 16 else 2
    ax.set_xticks(range(0, L, step))
    ax.set_yticks(range(0, L, step))
    ax.tick_params(labelsize=6)
    # outline the selected query row
    ax.add_patch(plt.Rectangle((-0.5, q - 0.5), L, 1, fill=False, edgecolor="red", lw=1.6))
    t = tp[q]
    if t >= 0:
        ax.scatter([t], [q], marker="x", color="red", s=90, linewidths=2)
    ax.set_title(f"{cond['rule']} (depth {cond['depth']}) — red ✕ = true target", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_query_bar(attn, cond, q):
    L = attn.shape[0]
    q = min(max(int(q), 0), L - 1)
    row = attn[q]
    tp = cond["target_pos"]
    t = tp[q]
    colors = ["#bbbbbb"] * L
    if t >= 0:
        colors[t] = "#e63946"
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.bar(range(L), row, color=colors)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Key position")
    ax.set_ylabel("attn weight")
    tgt = "none (excluded)" if t < 0 else f"node {t}"
    on_t = row[t] if t >= 0 else 0.0
    ax.set_title(f"Query {q}: target = {tgt}   (attn on target = {on_t:.3f})", fontsize=9)
    fig.tight_layout()
    return fig


def fig_ablation(meta):
    abl = meta["ablation"]
    keys = ["full", "addr_ablated", "scrambled", "uniform_baseline"]
    labels = ["full\ncircuit", "addr\nablated", "scrambled\ncodes", "uniform\nbaseline"]
    vals = [abl.get(k, 0.0) for k in keys]
    colors = ["#2a9d8f", "#e9c46a", "#f4a261", "#cccccc"]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("canonical correct-attn")
    ax.set_title("Causal ablation: the head needs the address codes", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def fig_sweep(bench):
    payload = (bench or {}).get("payload", {})
    sweep = payload.get("sweep", [])
    if not sweep:
        fig, ax = plt.subplots(figsize=(5.6, 3.0))
        ax.text(0.5, 0.5, "no sweep data", ha="center")
        ax.axis("off")
        return fig
    labels = [f"d{r['depth']}\n{r['path_rule'][:9]}" for r in sweep]
    means = [r["correct_attn_mean"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(labels, means, color="#264653")
    ax.axhline(1 / 14, color="red", ls="--", lw=1, label="uniform baseline ≈0.071")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("correct-attn mean")
    ax.set_title("Operating range: depth & path-rule sweep", fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def fmt_metrics(bench):
    if not bench:
        return "no benchmark.json"
    metrics = bench.get("metrics")
    if not metrics:
        return "no metrics in benchmark.json"
    keys = [
        "tree_path_canonical",
        "tree_path_depth_2",
        "tree_path_depth_4",
        "tree_path_ancestor_2",
        "tree_path_descendant",
        "tree_path_sibling",
        "tree_path_robustness",
        "tree_path_head_gap",
        "linear_baseline_canonical",
        "lift_over_baseline",
    ]
    lines = []
    for k in keys:
        if k in metrics:
            lines.append(f"{k:28s}: {metrics[k]:.4f}")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# update callbacks
# ──────────────────────────────────────────────────────────────────────
def refresh(run, key, q):
    meta = load_meta(run)
    if meta is None:
        empty = plt.figure()
        return empty, empty, empty, empty, "no run selected"
    if key not in cond_keys(meta):
        key = meta["canonical_key"]
    cond = get_cond(meta, key)
    attn = load_attn(run, key)
    bench = load_benchmark(run)
    if attn is None:
        empty = plt.figure()
        return empty, empty, fig_ablation(meta), fig_sweep(bench), fmt_metrics(bench)
    return (
        fig_heatmap(attn, cond, q),
        fig_query_bar(attn, cond, q),
        fig_ablation(meta),
        fig_sweep(bench),
        fmt_metrics(bench),
    )


def on_run_change(run):
    meta = load_meta(run)
    keys = cond_keys(meta)
    default_key = meta["canonical_key"] if meta else None
    return gr.update(choices=keys, value=default_key)


_runs = get_runs()
_init_run = _runs[0] if _runs else None
_init_meta = load_meta(_init_run)
_init_keys = cond_keys(_init_meta)
_init_key = _init_meta["canonical_key"] if _init_meta else None


with gr.Blocks(title="Attention Tree Path — pass_2") as demo:
    gr.Markdown(
        "# Attention Tree Path — pass_2 (hand-built QK content-addressing circuit)\n"
        "A single attention layer (no MLP) traces tree paths by **content lookup**: "
        "the query emits the *address code of the node it wants* and the dot product "
        "with each key's *own address code* resolves the unique target. "
        "Fixed, depth-independent Wq/Wk — the same head works at depth 2/3/4."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=_runs, value=_init_run, label="Run")
            cond_dd = gr.Dropdown(choices=_init_keys, value=_init_key, label="Condition (depth / path-rule)")
            q_slider = gr.Slider(0, 30, value=0, step=1, label="Query node (row)")
        with gr.Row():
            heatmap = gr.Plot(label="Attention heatmap")
            qbar = gr.Plot(label="Selected-query attention")
        with gr.Row():
            abl_plot = gr.Plot(label="Causal ablation")
            sweep_plot = gr.Plot(label="Operating-range sweep")
        metrics_box = gr.Textbox(label="Benchmark metrics", lines=11, interactive=False)

        run_dd.change(on_run_change, inputs=[run_dd], outputs=[cond_dd]).then(
            refresh,
            inputs=[run_dd, cond_dd, q_slider],
            outputs=[heatmap, qbar, abl_plot, sweep_plot, metrics_box],
        )
        cond_dd.change(
            refresh,
            inputs=[run_dd, cond_dd, q_slider],
            outputs=[heatmap, qbar, abl_plot, sweep_plot, metrics_box],
        )
        q_slider.change(
            refresh,
            inputs=[run_dd, cond_dd, q_slider],
            outputs=[heatmap, qbar, abl_plot, sweep_plot, metrics_box],
        )
        demo.load(
            refresh,
            inputs=[run_dd, cond_dd, q_slider],
            outputs=[heatmap, qbar, abl_plot, sweep_plot, metrics_box],
        )

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
