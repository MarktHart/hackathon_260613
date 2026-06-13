"""Gradio app for attention_bipartite / pass_2.

Demo tab: three views that together make the claim legible —
  (1) attention heatmaps  : baseline attends the diagonal (within-group);
                            the bipartite mask attends clean off-diagonal blocks.
  (2) bipartite-score bars: mechanism vs baseline across the num_heads sweep.
  (3) generalisation gap  : content-only learned attention overfits one batch
                            (train high, held-out 0); the positional mask
                            transfers to a fresh seed.
Benchmark tab: the shared cross-attempt leaderboard.
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
RESULTS = ATTEMPT_DIR / "results"


def list_runs():
    if not RESULTS.exists():
        return []
    return sorted((p.name for p in RESULTS.iterdir() if (p / "summary.json").exists()), reverse=True)


def load_run(run_id):
    d = RESULTS / run_id
    summary = json.loads((d / "summary.json").read_text())
    npz = np.load(d / "attn_example.npz")
    return summary, npz


# ---- figures -------------------------------------------------------------- #
def fig_heatmaps(summary, npz):
    gs = int(npz["group_size"])
    a_base, a_mech = npz["attn_baseline"], npz["attn_mechanism"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    for ax, mat, title in (
        (axes[0], a_base, "Baseline (no mask)\n→ attends self / within-group"),
        (axes[1], a_mech, "Bipartite mask (ours)\n→ attends cross-group only"),
    ):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=max(0.25, mat.max()))
        ax.axhline(gs - 0.5, color="w", lw=1.2)
        ax.axvline(gs - 0.5, color="w", lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("key position"); ax.set_ylabel("query position")
        ax.set_xticks([gs / 2 - 0.5, gs + gs / 2 - 0.5]); ax.set_xticklabels(["A", "B"])
        ax.set_yticks([gs / 2 - 0.5, gs + gs / 2 - 0.5]); ax.set_yticklabels(["A", "B"])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Attention weights (sample 0, 1 head)", fontsize=11)
    fig.tight_layout()
    return fig


def fig_bipartite_bars(summary):
    heads = [r["num_heads"] for r in summary["sweep_mechanism"]]
    mech = [r["mean_attn_between"] - r["mean_attn_within"] for r in summary["sweep_mechanism"]]
    base = [r["mean_attn_between"] - r["mean_attn_within"] for r in summary["sweep_baseline"]]
    x = np.arange(len(heads)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, mech, w, label="bipartite mask (ours)", color="#2a9d8f")
    ax.bar(x + w / 2, base, w, label="baseline (no mask)", color="#bbbbbb")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{h}h" for h in heads])
    ax.set_xlabel("num_heads"); ax.set_ylabel("bipartite score  (between − within)")
    ax.set_title("Bipartite score across head sweep  (bigger = more cross-group)")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_generalisation(summary):
    curve = summary["training_content_only"]
    steps = [c["step"] for c in curve]
    tr = [c["train_acc"] for c in curve]
    te = [c["heldout_acc"] for c in curve]
    g = summary["generalisation"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(steps, tr, "-o", ms=3, color="#e76f51", label="content-only: TRAIN seeds")
    ax.plot(steps, te, "-o", ms=3, color="#264653", label="content-only: HELD-OUT seed")
    ax.axhline(g["mask_heldout_acc"], color="#2a9d8f", ls="--",
               label=f"bipartite mask: held-out ({g['mask_heldout_acc']:.2f})")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="retrieval ceiling (0.5)")
    ax.set_xlabel("training step"); ax.set_ylabel("retrieval accuracy")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("Content attention OVERFITS; only the positional mask generalises")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    return fig


def summary_md(summary):
    c, g = summary["canonical"], summary["generalisation"]
    return (
        f"**Canonical (num_heads=4).** bipartite score: "
        f"`{c['bipartite_score_mechanism']:.3f}` (mask) vs `{c['bipartite_score_baseline']:.3f}` (baseline). "
        f"Retrieval: `{c['retrieval_mechanism']:.3f}` vs `{c['retrieval_baseline']:.3f}` "
        f"(ceiling `{c['retrieval_ceiling']}` — 2 same-feature keys per group).\n\n"
        f"**Held-out seed.** mask `{g['mask_heldout_acc']:.3f}`  ·  "
        f"content-only train `{g['content_only_train_acc']:.3f}` → held-out "
        f"`{g['content_only_heldout_acc']:.3f}`  ·  baseline `{g['baseline_heldout_acc']:.3f}`. "
        f"Content attention memorises noise; it cannot express the positional bipartite rule."
    )


def render(run_id):
    if not run_id:
        return None, None, None, "No runs found. Run `main.py` first."
    summary, npz = load_run(run_id)
    return (fig_heatmaps(summary, npz), fig_bipartite_bars(summary),
            fig_generalisation(summary), summary_md(summary))


with gr.Blocks(title="attention_bipartite · pass_2") as demo:
    gr.Markdown(
        "# Bipartite attention is a *positional* circuit\n"
        "Tokens carry **content only** (`q=k=v=feature_base+noise`). Standard content "
        "attention therefore attends to the self/within-group token and **fails** the "
        "cross-group task. A fixed **cross-group mask** (a one-line delta from "
        "`base_model.Attention`'s causal mask) produces robust bipartite attention; a "
        "learned content-only projection can only *overfit* one batch and gets **0** on a "
        "held-out seed."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=list_runs(), value=(list_runs() or [None])[0],
                                 label="run", scale=4)
            refresh = gr.Button("↻ refresh", scale=1)
        info = gr.Markdown()
        heat = gr.Plot(label="1 · attention weights: baseline vs bipartite mask")
        with gr.Row():
            bars = gr.Plot(label="2 · bipartite score across head sweep")
            gen = gr.Plot(label="3 · generalisation: overfit vs transfer")

        run_dd.change(render, inputs=run_dd, outputs=[heat, bars, gen, info])
        refresh.click(lambda: gr.update(choices=list_runs()), outputs=run_dd)
        demo.load(render, inputs=run_dd, outputs=[heat, bars, gen, info])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
