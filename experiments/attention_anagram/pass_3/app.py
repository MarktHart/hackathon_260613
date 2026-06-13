"""Gradio app — attention_anagram / pass_3 (hand-built token-identity head).

Demo tab makes the claim legible in three panels:
  1. Bar chart of alignment-on-true-source per permutation type for the
     hand-built MATCH circuit vs the POSITIONAL strawman vs the ABLATED control
     (QK knocked out) vs the dashed uniform baseline. This is the benchmarked
     metric plus the causal control: zero the QK circuit and alignment collapses
     to baseline -> the head *uses* token identity, it isn't an artefact.
  2. The hand-set QK matrix over the vocabulary (beta * Identity): the diagonal
     IS the mechanism — a target token attends the source token of the same id.
  3. Operating range over two axes: sequence length (2..256, the circuit holds
     because it has no positional term) and vocabulary size (the honest failure
     mode — small vocab -> token collisions -> attention splits), each against
     its analytic reference.
Benchmark tab: cross-attempt history.
"""
import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"


def list_runs():
    if not RESULTS.exists():
        return []
    runs = sorted(p.name for p in RESULTS.iterdir() if (p / "diag.json").exists())
    return runs[::-1]


def load_diag(run_id):
    if not run_id:
        return None
    p = RESULTS / run_id / "diag.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def summary(run_id):
    d = load_diag(run_id)
    if d is None:
        return "No run found. Run main.py first."
    c = d["canonical"]
    return (
        f"**Canonical condition — random permutation, L=8, vocab=50**  \n"
        f"**Hand-built MATCH circuit:** {c['match']:.4f}  \n"
        f"**POSITIONAL strawman (ignores tokens):** {c['positional']:.4f}  \n"
        f"**ABLATED (QK knocked out):** {c['ablated']:.4f}  \n"
        f"**Uniform baseline:** {d['uniform_baseline']:.4f}  \n\n"
        f"The match circuit puts almost all attention on the true source position; "
        f"the strawman and the ablated control both collapse to the uniform "
        f"baseline — so the lift comes from *token-identity matching*, the "
        f"hand-set diagonal QK matrix shown in the middle panel."
    )


def plot_run(run_id):
    d = load_diag(run_id)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    if d is None:
        axes[0].text(0.5, 0.5, "No run", ha="center")
        return fig

    # --- 1. match vs positional vs ablated vs uniform, per perm type ---
    ax = axes[0]
    perm_types = ["swap", "rotation", "random"]
    bars = d["perm_bars"]
    x = np.arange(len(perm_types))
    w = 0.26
    ax.bar(x - w, [bars["match"].get(p, 0.0) for p in perm_types], w,
           label="match (hand-built)", color="#2b6cb0")
    ax.bar(x, [bars["positional"].get(p, 0.0) for p in perm_types], w,
           label="positional strawman", color="#ed8936")
    ax.bar(x + w, [bars["ablated"].get(p, 0.0) for p in perm_types], w,
           label="ablated (QK->0)", color="#cbd5e0")
    ax.axhline(d["uniform_baseline"], color="crimson", ls="--",
               label=f"uniform ({d['uniform_baseline']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(perm_types)
    ax.set_ylim(0, 1.05); ax.set_ylabel("alignment on true source pos")
    ax.set_title("Token matcher vs strawman vs ablation")
    ax.legend(fontsize=8, loc="center right")

    # --- 2. hand-set QK matrix over the vocab (diagonal = the mechanism) ---
    ax = axes[1]
    M = np.array(d["qk_matrix"])
    im = ax.imshow(M, cmap="magma")
    ax.set_xlabel("source token id"); ax.set_ylabel("target token id")
    ax.set_title("Hand-set QK matrix = beta * I\n(diagonal = token-identity match)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- 3. operating range: seq_len (holds) and vocab (collision failure) ---
    ax = axes[2]
    s = d["op_seq"]
    ax.plot(s["seq_lens"], s["alignment"], marker="o", color="#2b6cb0",
            label="match vs seq_len")
    ax.plot(s["seq_lens"], s["baseline"], marker="x", ls="--", color="crimson",
            label="uniform 1/L")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length (log2)  /  vocab size (log2)")
    ax.set_ylabel("mean alignment")
    ax.set_ylim(0, 1.05)
    v = d["op_vocab"]
    ax.plot(v["vocabs"], v["alignment"], marker="s", color="#38a169",
            label="match vs vocab")
    ax.plot(v["vocabs"], v["expected"], marker=".", ls=":", color="#276749",
            label="vocab analytic E[1/copies]")
    ax.set_title("Operating range: seq_len holds, small vocab degrades")
    ax.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    return fig


with gr.Blocks(title="attention_anagram / pass_3") as demo:
    gr.Markdown("# attention_anagram — a *hand-built* token-identity matching head")
    gr.Markdown(
        "One cross-attention layer (no MLP, **no positional term**): query = "
        "`E[target]`, key = `E[source]`, with the QK circuit hand-set to the "
        "identity over the vocabulary. Each target token attends the source "
        "position holding the **same token id** — solving the anagram alignment "
        "task with zero training. The strawman and the QK ablation collapse to "
        "the uniform baseline."
    )

    with gr.Tab("Demo"):
        runs = list_runs()
        run_dd = gr.Dropdown(runs, value=(runs[0] if runs else None),
                             label="Run (newest first)")
        summary_md = gr.Markdown()
        run_plot = gr.Plot()

        def refresh(run_id):
            return plot_run(run_id), summary(run_id)

        run_dd.change(refresh, inputs=run_dd, outputs=[run_plot, summary_md])

        def init():
            r = list_runs()
            rid = r[0] if r else None
            return plot_run(rid), summary(rid)

        demo.load(init, inputs=None, outputs=[run_plot, summary_md])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
