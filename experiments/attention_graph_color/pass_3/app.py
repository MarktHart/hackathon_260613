"""Gradio app for attention_graph_color / pass_3.

Demo tab visualises the claim three ways:
  1. Ablation bar chart  — full vs color-only vs edge-only(~pass_2) vs uniform.
     This is the headline: the colour term, not the adjacency term, produces
     the separation.
  2. Attention heatmap   — nodes sorted by colour; off-diagonal colour blocks
     (different colours) light up, on-diagonal blocks (same colour) stay dark.
  3. Operating range     — color_separation vs graph size n (20..320).

Benchmark tab drops in the canonical cross-attempt panel.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent


def _latest_run() -> Path | None:
    runs = sorted((ATTEMPT_DIR / "results").glob("*"))
    runs = [r for r in runs if (r / "benchmark.json").exists()]
    return runs[-1] if runs else None


def _read_json(run: Path, name: str):
    p = run / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _empty_fig(msg: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


# ---------------------------------------------------------------------------
# 1. Ablation bar chart
# ---------------------------------------------------------------------------
def ablation_fig():
    run = _latest_run()
    if run is None:
        return _empty_fig("No run yet — execute main.py first.")
    abl = _read_json(run, "ablation.json")
    if not abl:
        return _empty_fig("ablation.json missing.")

    order = ["full (color+edge)", "color-only", "edge-only (~pass_2)", "uniform baseline"]
    order = [k for k in order if k in abl]
    sep = [abl[k]["color_separation_canonical"] for k in order]
    edge = [abl[k]["edge_respect_canonical"] for k in order]

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    b1 = ax.bar(x - w / 2, sep, w, label="color_separation (all pairs)", color="#3b7dd8")
    b2 = ax.bar(x + w / 2, edge, w, label="edge_respect (edges only)", color="#e08a2b")
    ax.axhline(0, color="#888", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=12, ha="right", fontsize=9)
    ax.set_ylabel("attention metric (n=40)")
    ax.set_title("Ablation: the COLOR term drives separation, edge term adds edge-respect")
    ax.legend(fontsize=8)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:.3f}", (r.get_x() + r.get_width() / 2, h),
                        ha="center", va="bottom" if h >= 0 else "top", fontsize=7)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Attention heatmap (nodes sorted by colour)
# ---------------------------------------------------------------------------
def _samples(run: Path):
    p = run / "samples.npz"
    if not p.exists():
        return None
    return np.load(p)


def sample_choices():
    run = _latest_run()
    if run is None:
        return []
    meta = _read_json(run, "samples_meta.json") or []
    return [m["label"] for m in meta]


def heatmap_fig(label: str):
    run = _latest_run()
    if run is None:
        return _empty_fig("No run yet.")
    meta = _read_json(run, "samples_meta.json") or []
    data = _samples(run)
    if data is None or not meta:
        return _empty_fig("samples.npz missing.")
    idx = 0
    for i, m in enumerate(meta):
        if m["label"] == label:
            idx = i
            break

    attn = data[f"attn_{idx}"]
    colors = data[f"colors_{idx}"]
    order = np.argsort(colors, kind="stable")
    A = attn[order][:, order]
    sc = colors[order]
    bounds = np.where(np.diff(sc) != 0)[0] + 0.5  # block boundaries

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(A, cmap="magma", interpolation="nearest")
    for b in bounds:
        ax.axhline(b, color="#39ff88", lw=0.8, alpha=0.7)
        ax.axvline(b, color="#39ff88", lw=0.8, alpha=0.7)
    ax.set_title("Attention, nodes sorted by colour\n"
                 "green lines = colour blocks; diagonal blocks (same colour) stay dark")
    ax.set_xlabel("attended-to node (key)")
    ax.set_ylabel("attending node (query)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()
    return fig


def heatmap_caption(label: str):
    run = _latest_run()
    if run is None:
        return "No run yet."
    meta = _read_json(run, "samples_meta.json") or []
    for m in meta:
        if m["label"] == label:
            return (f"**{m['label']}** — per-graph color_separation = "
                    f"`{m['color_separation']:.4f}`. Bright off-block cells are "
                    f"attention to *differently*-coloured nodes; dark diagonal "
                    f"blocks are same-coloured nodes that get starved.")
    return ""


# ---------------------------------------------------------------------------
# 3. Operating range
# ---------------------------------------------------------------------------
def range_fig():
    run = _latest_run()
    if run is None:
        return _empty_fig("No run yet.")
    ext = _read_json(run, "extended_range.json")
    if not ext:
        return _empty_fig("extended_range.json missing.")
    ns = [e["n"] for e in ext]
    sep = [e["color_separation"] for e in ext]
    std = [e.get("std", 0.0) for e in ext]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.errorbar(ns, sep, yerr=std, marker="o", color="#3b7dd8", capsize=3)
    ax.axhline(0, color="#888", lw=0.8, ls="--", label="uniform baseline ≈ 0")
    ax.set_xscale("log")
    ax.set_xlabel("graph size n  (log scale, p=0.2)")
    ax.set_ylabel("color_separation")
    ax.set_title("Operating range: separation stays positive as n grows ~16x")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def headline_md():
    run = _latest_run()
    if run is None:
        return "**No run yet** — run `main.py`."
    bj = _read_json(run, "benchmark.json")
    abl = _read_json(run, "ablation.json") or {}
    if not bj:
        return "benchmark.json missing."
    metrics = bj.get("metrics", bj)  # record_benchmark may nest; fall back
    # benchmark.json from record_benchmark stores the payload+metrics; be tolerant
    sep = None
    if isinstance(metrics, dict):
        sep = metrics.get("color_separation_canonical")
    full = abl.get("full (color+edge)", {})
    edge_only = abl.get("edge-only (~pass_2)", {})
    parts = [f"### Latest run: `{run.name}`"]
    if full:
        parts.append(f"- **full mechanism** color_separation = "
                     f"`{full['color_separation_canonical']:.4f}`, "
                     f"edge_respect = `{full['edge_respect_canonical']:.4f}`")
    if edge_only:
        parts.append(f"- **edge-only (≈ pass_2)** color_separation = "
                     f"`{edge_only['color_separation_canonical']:.4f}`  ← the colour "
                     f"term is what closes this gap")
    return "\n".join(parts)


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_graph_color · pass_3\n"
        "A hand-built attention head where the score is "
        "`w_color·(colours differ) + w_adj·edge`, then row-softmax. "
        "The colour term — **not** an adjacency mask — produces the separation."
    )
    head_md = gr.Markdown()

    with gr.Tab("Demo"):
        gr.Markdown("## 1. Ablation — what actually does the work")
        gr.Markdown(
            "Zeroing the colour term (`edge-only`) reproduces the previous "
            "attempt's behaviour and collapses `color_separation`; zeroing the "
            "edge term keeps almost all separation. This is the faithfulness "
            "check: the colour computation is load-bearing, not redundant."
        )
        abl_plot = gr.Plot()

        gr.Markdown("## 2. Attention heatmap (nodes grouped by colour)")
        samp_dd = gr.Dropdown(label="sample graph", choices=[], interactive=True)
        heat_plot = gr.Plot()
        heat_cap = gr.Markdown()

        gr.Markdown("## 3. Operating range")
        range_plot = gr.Plot()

    with gr.Tab("Benchmark"):
        gr.Markdown("Cross-attempt leaderboard and metric history.")
        panel = benchmark_panel(str(GOAL_DIR))
        if panel is not None:
            panel.render()
        else:
            gr.Markdown("_(benchmark panel unavailable — run an attempt first)_")

    def _init():
        choices = sample_choices()
        first = choices[0] if choices else None
        return (
            headline_md(),
            ablation_fig(),
            gr.update(choices=choices, value=first),
            heatmap_fig(first) if first else _empty_fig("no samples"),
            heatmap_caption(first) if first else "",
            range_fig(),
        )

    demo.load(_init, inputs=None,
              outputs=[head_md, abl_plot, samp_dd, heat_plot, heat_cap, range_plot])
    samp_dd.change(heatmap_fig, inputs=samp_dd, outputs=heat_plot)
    samp_dd.change(heatmap_caption, inputs=samp_dd, outputs=heat_cap)


if __name__ == "__main__":
    demo.launch()
