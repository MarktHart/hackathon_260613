"""Gradio app — attention_anagram / pass_2 (trained head).

Demo tab makes three claims legible:
  1. The TRAINED head aligns target->true-source far above the untrained
     same-architecture strawman and the uniform baseline (bar chart).
  2. It learned token-identity matching: the effective QK matrix over the
     vocab is diagonal (heatmap) — the interp / faithfulness evidence.
  3. Because there is no positional component, the learned circuit transfers
     across sequence length over 2+ orders of magnitude (line plot).
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
SEQ = 8


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
    return (f"**Trained canonical alignment:** {d['trained_canonical']:.4f}  \n"
            f"**Untrained (same arch) strawman:** {d['untrained_canonical']:.4f}  \n"
            f"**Uniform baseline:** {d['uniform_baseline']:.4f}  \n"
            f"**Token-match matrix** — diagonal mean {d['tm_diag_mean']:.2f} vs "
            f"off-diagonal mean {d['tm_offdiag_mean']:.2f} "
            f"(large gap means the head matches tokens by identity).")


def plot_run(run_id):
    d = load_diag(run_id)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    if d is None:
        axes[0].text(0.5, 0.5, "No run", ha="center")
        return fig

    # --- 1. trained vs untrained vs baseline, per perm type ---
    ax = axes[0]
    perm_types = ["swap", "rotation", "random"]
    tr = [d["perm_bars"].get(p, 0.0) for p in perm_types]
    un = [d["untrained_perm_bars"].get(p, 0.0) for p in perm_types]
    x = np.arange(len(perm_types))
    ax.bar(x - 0.2, tr, 0.4, label="trained", color="#2b6cb0")
    ax.bar(x + 0.2, un, 0.4, label="untrained (strawman)", color="#cbd5e0")
    ax.axhline(d["uniform_baseline"], color="crimson", ls="--",
               label=f"uniform ({d['uniform_baseline']:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(perm_types)
    ax.set_ylim(0, 1.05); ax.set_ylabel("alignment on true source pos")
    ax.set_title("Trained head vs strawman"); ax.legend(fontsize=8)

    # --- 2. learned token-match matrix (should be diagonal) ---
    ax = axes[1]
    M = np.array(d["token_match_matrix"])
    im = ax.imshow(M, cmap="magma")
    ax.set_xlabel("source token id"); ax.set_ylabel("target token id")
    ax.set_title("Learned QK matrix M[a,b]\n(diagonal = token-identity matching)")
    fig.colorbar(im, ax=ax, fraction=0.046)

    # --- 3. operating range over seq_len ---
    ax = axes[2]
    op = d["op_range"]
    ax.plot(op["seq_lens"], op["alignment"], marker="o", color="#2b6cb0", label="trained head")
    ax.plot(op["seq_lens"], op["baseline"], marker="x", ls="--", color="crimson", label="uniform 1/L")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length (log2)"); ax.set_ylabel("mean alignment")
    ax.set_ylim(0, 1.05); ax.set_title("Operating range (trained at L=8)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


with gr.Blocks(title="attention_anagram / pass_2") as demo:
    gr.Markdown("# attention_anagram — a *trained* token-matching attention head")
    gr.Markdown(
        "One attention-only layer (no MLP, no positional embedding), 8 heads, "
        "**trained** to align each target token to its true source position. "
        "It discovers token-identity matching on its own — shown by the diagonal "
        "learned QK matrix — and the circuit transfers across sequence length."
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
