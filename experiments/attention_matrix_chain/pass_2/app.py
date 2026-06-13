"""Gradio app for attention_matrix_chain / pass_2.

Demo tab: the virtual-attention-head circuit made visible.
  * heatmaps of A1, A2, the circuit's reconstructed A_chain, and the
    ground-truth A_chain — for any alpha in the sweep, so a human can
    eyeball that the composed pattern matches;
  * an ablation chart showing both layers are causally necessary: knocking
    out layer 2 (predict A1) or layer 1 (predict A2 = single-hop) collapses
    fidelity in the peaked regime, while the full circuit stays ~1.0.

Benchmark tab: the shared cross-attempt panel.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

_HERE = Path(__file__).resolve().parent
_RESULTS = _HERE / "results"
_GOAL_DIR = str(_HERE.parent)


def _list_runs():
    if not _RESULTS.exists():
        return []
    runs = [p.name for p in _RESULTS.iterdir()
            if p.is_dir() and (p / "examples.npz").exists()]
    return sorted(runs, reverse=True)


def _load_run(run):
    if not run:
        return None
    rd = _RESULTS / run
    ex_path = rd / "examples.npz"
    ab_path = rd / "ablation.json"
    if not ex_path.exists():
        return None
    ex = np.load(ex_path)
    ablation = None
    if ab_path.exists():
        with open(ab_path) as f:
            ablation = json.load(f)
    return {
        "alphas": ex["alphas"],
        "A1": ex["A1"], "A2": ex["A2"],
        "pred": ex["pred"], "true": ex["true"],
        "ablation": ablation,
    }


def _alpha_choices(run):
    data = _load_run(run)
    if data is None:
        return []
    return [f"{a:.1f}" for a in data["alphas"].tolist()]


def _default_alpha(choices):
    if not choices:
        return None
    return choices[1] if len(choices) > 1 else choices[0]


def render_heatmaps(run, alpha_label):
    data = _load_run(run)
    if data is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No run found.\nRun main.py first.",
                ha="center", va="center")
        ax.axis("off")
        return fig

    alphas = [f"{a:.1f}" for a in data["alphas"].tolist()]
    ai = alphas.index(alpha_label) if alpha_label in alphas else 0

    A1, A2 = data["A1"][ai], data["A2"][ai]
    pred, true = data["pred"][ai], data["true"][ai]
    err = np.abs(pred - true)

    panels = [
        ("A1  (layer-1 pattern)", A1, "viridis"),
        ("A2  (layer-2 pattern)", A2, "viridis"),
        ("circuit  A_chain = A2@A1", pred, "viridis"),
        ("ground truth A_chain", true, "viridis"),
        (f"|pred - true|  (max {err.max():.1e})", err, "magma"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.4))
    for ax, (title, M, cmap) in zip(axes, panels):
        is_err = title.startswith("|pred")
        vmax = (M.max() if M.max() > 0 else 1.0) if is_err else 1.0
        im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("key pos")
        ax.set_ylabel("query pos")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Head 0, alpha = {alpha_label}  —  two stacked attention layers "
        f"write the composed pattern into the residual stream",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def render_ablation(run):
    data = _load_run(run)
    if data is None or data["ablation"] is None:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No ablation data.", ha="center", va="center")
        ax.axis("off")
        return fig

    ab = data["ablation"]
    alphas = ab["alpha_sweep"]
    x = np.arange(len(alphas))

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(x, ab["full_circuit"], "o-", lw=2.5, color="#1b9e77",
            label="full circuit (both layers)")
    ax.plot(x, ab["ablate_layer2_predicts_A1"], "s--", color="#d95f02",
            label="ablate layer 2  -> predict A1")
    ax.plot(x, ab["ablate_layer1_predicts_A2"], "^--", color="#7570b3",
            label="ablate layer 1  -> predict A2 (single-hop)")
    ax.plot(x, ab["single_hop_baseline"], "x:", color="#999999",
            label="single-hop baseline (A2)")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_xlabel("Dirichlet alpha  (left = peaked rows, composition matters)")
    ax.set_ylabel("chain fidelity (1 - mean row TV)")
    ax.set_ylim(0, 1.02)
    ax.invert_xaxis()  # peaked / hard regime on the left visually
    ax.set_title("Causal ablation: removing EITHER layer breaks composition")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention Matrix Chain — virtual attention head circuit\n"
        "Two stacked attention layers with hand-set identity OV weights "
        "compose into the two-hop pattern `A_chain = A2 @ A1`, written into "
        "the residual stream. The Demo tab shows the reconstruction and a "
        "causal ablation that proves both layers are necessary."
    )

    with gr.Tab("Demo"):
        _runs = _list_runs()
        _default_run = _runs[0] if _runs else None
        _init_choices = _alpha_choices(_default_run)
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=_runs, value=_default_run, label="results run",
                scale=2)
            alpha_dd = gr.Dropdown(
                choices=_init_choices, value=_default_alpha(_init_choices),
                label="Dirichlet alpha (peakedness)", scale=1)
        gr.Markdown(
            "**Heatmaps** — the circuit's `A_chain` (panel 3) should match "
            "ground truth (panel 4); the error panel (5) should be ~0. Note "
            "how at small alpha the composed pattern (3/4) looks nothing like "
            "`A2` alone (panel 2) — that is the regime the single-hop shortcut "
            "fails and composition is essential."
        )
        heat = gr.Plot(label="A1 / A2 / circuit A_chain / truth / error")
        gr.Markdown(
            "**Ablation** — knock out layer 2 and the circuit can only emit "
            "`A1`; knock out layer 1 and it collapses to `A2` (the single-hop "
            "baseline). Only the full two-layer circuit stays near 1.0 as "
            "rows become peaked."
        )
        abl = gr.Plot(label="ablation fidelity across alpha")

        def _refresh(run, alpha):
            return render_heatmaps(run, alpha), render_ablation(run)

        def _on_run_change(run):
            choices = _alpha_choices(run)
            val = _default_alpha(choices)
            return (gr.update(choices=choices, value=val),
                    render_heatmaps(run, val), render_ablation(run))

        run_dd.change(_on_run_change, inputs=run_dd,
                      outputs=[alpha_dd, heat, abl])
        alpha_dd.change(render_heatmaps, inputs=[run_dd, alpha_dd],
                        outputs=heat)
        demo.load(_refresh, inputs=[run_dd, alpha_dd], outputs=[heat, abl])

    with gr.Tab("Benchmark"):
        benchmark_panel(_GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
