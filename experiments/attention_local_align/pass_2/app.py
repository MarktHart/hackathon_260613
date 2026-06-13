"""Gradio app for attention_local_align / pass_2.

Demo tab: shows that a single QK attention head, built from positional
encodings + a rotation-by-delta in W_Q, *computes* predecessor selection.
Four views make the claim checkable:
  1. offset x data-shift matrix   — diagonal => each rotated head aligns to its shift
  2. canonical sweep + baselines  — predecessor head wins at shift=-1, ~0 elsewhere
  3. faithfulness ablations       — kill positions/rotation => alignment collapses;
                                    kill/shuffle tokens => unchanged (head is positional)
  4. operating range over T       — alignment across 8..512 token sequences
Plus an example attention heatmap (the sub-diagonal band).

Benchmark tab: the shared cross-attempt panel.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"


def list_runs():
    if not RESULTS.exists():
        return []
    runs = [p.name for p in RESULTS.glob("*") if (p / "analysis.json").exists()]
    return sorted(runs, reverse=True)


def _load(run):
    if run is None:
        runs = list_runs()
        if not runs:
            return None
        run = runs[0]
    p = RESULTS / run / "analysis.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _empty(msg):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


def plot_matrix(run):
    a = _load(run)
    if a is None:
        return _empty("No runs yet")
    M = a["offset_shift_matrix"]
    offsets = a["offsets"]
    shifts = a["data_shifts"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(shifts)))
    ax.set_xticklabels([f"{s:+d}" for s in shifts])
    ax.set_yticks(range(len(offsets)))
    ax.set_yticklabels([f"{o:+d}" for o in offsets])
    ax.set_xlabel("data ground-truth shift")
    ax.set_ylabel("head rotation  δ  (W_Q)")
    ax.set_title("alignment(δ, shift) — bright diagonal = QK selects shift=δ")
    for i in range(len(offsets)):
        for j in range(len(shifts)):
            ax.text(j, i, f"{M[i][j]:.2f}", ha="center", va="center",
                    color="white" if M[i][j] < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="mean max attn on target")
    fig.tight_layout()
    return fig


def plot_sweep(run):
    a = _load(run)
    if a is None:
        return _empty("No runs yet")
    sw = a["model_sweep"]
    rnd = {r["shift"]: r["align"] for r in a["random_sweep"]}
    shifts = [s["shift"] for s in sw]
    align = [s["align"] for s in sw]
    uniform = a["ablations"]["uniform_baseline"]
    colors = ["#1f77b4" if s == -1 else "#aec7e8" for s in shifts]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    x = range(len(shifts))
    ax.bar(x, align, color=colors, label="predecessor head (δ=-1)")
    ax.plot(x, [rnd.get(s, 0) for s in shifts], "x--", color="green",
            label="random-attn strawman")
    ax.axhline(uniform, color="red", ls=":", label=f"uniform 1/(T-1)≈{uniform:.3f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s:+d}" for s in shifts])
    ax.set_xlabel("data ground-truth shift")
    ax.set_ylabel("mean max attn on target")
    ax.set_title("predecessor head: ~1.0 at shift=-1, ~0 elsewhere")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_ablations(run):
    a = _load(run)
    if a is None:
        return _empty("No runs yet")
    ab = a["ablations"]
    order = [
        ("full", "full circuit"),
        ("zero_tokens", "zero tokens"),
        ("shuffle_tokens", "shuffle tokens"),
        ("identity_rotation", "δ=0 (no shift)"),
        ("zero_positions", "zero positions"),
        ("random_strawman", "random strawman"),
    ]
    labels = [lbl for k, lbl in order]
    vals = [ab[k] for k, _ in order]
    # blue = circuit intact, orange = circuit broken
    colors = ["#1f77b4", "#1f77b4", "#1f77b4", "#ff7f0e", "#ff7f0e", "#7f7f7f"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(labels, vals, color=colors)
    ax.axhline(ab["uniform_baseline"], color="red", ls=":",
               label=f"uniform≈{ab['uniform_baseline']:.3f}")
    ax.set_ylabel("canonical alignment (shift=-1)")
    ax.set_title("faithfulness: positions+rotation matter, tokens don't")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def plot_seqlen(run):
    a = _load(run)
    if a is None:
        return _empty("No runs yet")
    c = a["seqlen_curve"]
    Ts = [r["T"] for r in c]
    vals = [r["align"] for r in c]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(Ts, vals, "o-", color="#1f77b4")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("sequence length T (log2)")
    ax.set_ylabel("attn on predecessor (t-1)")
    ax.set_title("operating range: T = 8 … 512")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(run):
    a = _load(run)
    if a is None:
        return _empty("No runs yet")
    ex = a["example_attn"]
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(ex, cmap="magma", aspect="auto")
    ax.set_xlabel("key position s")
    ax.set_ylabel("query position t")
    ax.set_title("example attention — sub-diagonal band (s = t-1)")
    fig.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    return fig


def refresh(run):
    return (
        plot_matrix(run),
        plot_sweep(run),
        plot_ablations(run),
        plot_seqlen(run),
        plot_heatmap(run),
    )


with gr.Blocks() as demo:
    gr.Markdown("# attention_local_align — pass_2")
    gr.Markdown(
        "A single attention head where predecessor selection is **computed by "
        "the QK dot-product** over positional encodings (W_Q = rotation-by-δ, "
        "W_K = identity), not written into the output. δ=-1 ⇒ each query attends "
        "to token *t-1*."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=list_runs(),
                value=(list_runs()[0] if list_runs() else None),
                label="run",
            )
            refresh_btn = gr.Button("refresh")
        with gr.Row():
            m_plot = gr.Plot(label="offset × shift")
            s_plot = gr.Plot(label="canonical sweep")
        with gr.Row():
            ab_plot = gr.Plot(label="ablations")
            sl_plot = gr.Plot(label="operating range")
        with gr.Row():
            hm_plot = gr.Plot(label="example attention")

        outputs = [m_plot, s_plot, ab_plot, sl_plot, hm_plot]
        run_dd.change(refresh, inputs=run_dd, outputs=outputs)
        refresh_btn.click(refresh, inputs=run_dd, outputs=outputs)
        demo.load(refresh, inputs=run_dd, outputs=outputs)

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
