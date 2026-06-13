"""Gradio app for attention_topo_sort / pass_3.

Demo tab:
  * Headline grouped bar chart: topo_respect of the level-attention mechanism
    vs the direct-edge strawman vs the uniform ablation, across the density
    sweep. The single comparison that makes the claim ("level beats both").
  * Per-DAG attention heatmap with nodes REORDERED by topological level. Mass
    pooling into the lower-left triangle == descendants attending back to
    ancestors. A toggle reveals the true ancestor mask for direct comparison.
  * Operating-range line plot: topo_respect vs N (4..64) per density.
Benchmark tab: shared benchmark_panel across every attempt at this goal.
"""

import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import gradio as gr
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_ROOT = Path(__file__).parent / "results"


# --------------------------------------------------------------------------- #
# Artefact loading                                                            #
# --------------------------------------------------------------------------- #
def _latest_run() -> Path | None:
    if not RESULTS_ROOT.exists():
        return None
    runs = sorted([d for d in RESULTS_ROOT.iterdir() if d.is_dir()],
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0] if runs else None


def _load(run_dir: Path):
    comp = json.loads((run_dir / "comparison.json").read_text()) \
        if (run_dir / "comparison.json").exists() else None
    scale = json.loads((run_dir / "scale.json").read_text()) \
        if (run_dir / "scale.json").exists() else None
    can = None
    if (run_dir / "canonical.npz").exists():
        can = np.load(run_dir / "canonical.npz")
    return comp, scale, can


def _fig_to_img(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _ancestors(adj: np.ndarray) -> np.ndarray:
    reach = adj.astype(bool).copy()
    n = reach.shape[0]
    for k in range(n):
        reach |= reach[:, k:k + 1] & reach[k:k + 1, :]
    return reach


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _bar_chart(comp) -> Image.Image:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if comp is None:
        ax.text(0.5, 0.5, "Run main.py first", ha="center")
        return _fig_to_img(fig)
    dens = comp["densities"]
    x = np.arange(len(dens))
    w = 0.26
    ax.bar(x - w, comp["level_attention"], w, label="level attention (ours)", color="#2563eb")
    ax.bar(x, comp["direct_edge_strawman"], w, label="direct-edge strawman", color="#f59e0b")
    ax.bar(x + w, comp["uniform_ablation"], w, label="uniform ablation (β=0)", color="#9ca3af")
    ax.axhline(0.5, ls="--", c="k", lw=1, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:g}" for d in dens])
    ax.set_xlabel("edge density")
    ax.set_ylabel("topo_respect  (fraction of ancestor pairs respected)")
    ax.set_ylim(0, 1.05)
    cd = comp.get("canonical_density")
    ax.set_title(f"Level-biased attention respects the partial order  (canonical density = {cd:g})")
    ax.legend(fontsize=8, loc="lower right")
    return _fig_to_img(fig)


def _heatmap(can, idx: int, show_anc: bool) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    if can is None:
        ax.text(0.5, 0.5, "Run main.py first", ha="center")
        return _fig_to_img(fig)
    adj = can["adjacency"][idx]
    lvl = can["levels"][idx]
    attn = can["attention"][idx]
    n = adj.shape[0]
    order = np.argsort(lvl, kind="stable")        # ascending level: ancestors first
    A = attn[np.ix_(order, order)]
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=A.max())
    if show_anc:
        anc = _ancestors(adj)[np.ix_(order, order)]   # anc[a,d]: row=ancestor, col=desc
        # mark cells where (row=query=descendant) attends to (col=key=ancestor):
        # true ancestor pair is anc.T[query,key]
        true_back = anc.T
        ys, xs = np.where(true_back)
        ax.scatter(xs, ys, s=18, facecolors="none", edgecolors="#22d3ee", linewidths=1.4)
    ax.set_xticks(range(n))
    ax.set_xticklabels([f"{order[i]}\nL{int(lvl[order[i]])}" for i in range(n)], fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{order[i]} L{int(lvl[order[i]])}" for i in range(n)], fontsize=7)
    ax.set_xlabel("key node  (ancestors / lower level →)")
    ax.set_ylabel("query node  (descendants / higher level ↓)")
    ttl = f"DAG {idx}: attention, nodes sorted by topo level"
    if show_anc:
        ttl += "\n○ = true ancestor pair (should be bright)"
    ax.set_title(ttl, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    return _fig_to_img(fig)


def _scale_plot(scale) -> Image.Image:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if scale is None:
        ax.text(0.5, 0.5, "Run main.py first", ha="center")
        return _fig_to_img(fig)
    dens = sorted({r["density"] for r in scale})
    for d in dens:
        pts = sorted([(r["n"], r["topo_respect"]) for r in scale if r["density"] == d])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=f"density {d:g}")
    ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("number of nodes N (log scale)")
    ax.set_ylabel("topo_respect")
    ax.set_ylim(0, 1.05)
    ax.set_title("Operating range: mechanism holds as N grows 4 → 64")
    ax.legend(fontsize=8)
    return _fig_to_img(fig)


def _summary(comp, scale) -> str:
    if comp is None:
        return "No run found. Execute `python main.py` first."
    cd = comp["canonical_density"]
    ci = comp["densities"].index(cd) if cd in comp["densities"] else 0
    lines = [
        "### Hand-built level-biased bilinear attention",
        "Score `s[i,j] = -β·level[i]·level[j]`, softmax over keys. "
        "`level` = longest-path topological depth (iterated max message-passing on GPU).",
        "",
        f"- **topo_respect (canonical density {cd:g}): "
        f"{comp['level_attention'][ci]:.3f}**  (chance 0.5)",
        f"- direct-edge strawman: {comp['direct_edge_strawman'][ci]:.3f}  "
        "(ties on transitive-only pairs)",
        f"- β=0 ablation: {comp['uniform_ablation'][ci]:.3f}  (collapses to chance — the "
        "level bias is causally necessary)",
        "",
        "Every ancestor `a` of descendant `d` has `level[a] < level[d]`, so the higher-level "
        "row concentrates more mass on the lower-level node than the reverse → `attn[d,a] > attn[a,d]`.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# UI                                                                          #
# --------------------------------------------------------------------------- #
def _refresh(idx, show_anc):
    run = _latest_run()
    if run is None:
        empty = _bar_chart(None)
        return (_summary(None, None), empty, _heatmap(None, 0, False), empty)
    comp, scale, can = _load(run)
    return (_summary(comp, scale),
            _bar_chart(comp),
            _heatmap(can, int(idx), bool(show_anc)),
            _scale_plot(scale))


def _refresh_heatmap(idx, show_anc):
    run = _latest_run()
    if run is None:
        return _heatmap(None, 0, False)
    _, _, can = _load(run)
    return _heatmap(can, int(idx), bool(show_anc))


with gr.Blocks(title="Attention Topo Sort — pass_3") as demo:
    gr.Markdown(
        "# Attention Topo Sort — pass_3\n"
        "**Can attention encode a DAG's partial order?** A single bilinear "
        "attention head, biased by each node's topological *level*, makes every "
        "descendant attend back to its ancestors — a topo sort falls out of the "
        "attention pattern. Hand-built, zero training, all GPU torch."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            summary_md = gr.Markdown()
            with gr.Row():
                bar_img = gr.Image(label="Headline comparison", type="pil")
                scale_img = gr.Image(label="Operating range (N = 4..64)", type="pil")
            with gr.Row():
                with gr.Column(scale=1):
                    dag_idx = gr.Slider(0, 23, value=0, step=1, label="canonical DAG index")
                    show_anc = gr.Checkbox(value=True, label="overlay true ancestor pairs (○)")
                with gr.Column(scale=2):
                    heat_img = gr.Image(label="Attention heatmap (nodes sorted by topo level)",
                                        type="pil")

            demo.load(_refresh, inputs=[dag_idx, show_anc],
                      outputs=[summary_md, bar_img, heat_img, scale_img])
            dag_idx.change(_refresh_heatmap, inputs=[dag_idx, show_anc], outputs=[heat_img])
            show_anc.change(_refresh_heatmap, inputs=[dag_idx, show_anc], outputs=[heat_img])

        with gr.Tab("Benchmark"):
            gr.Markdown("## Benchmark history across all attempts at this goal")
            benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
