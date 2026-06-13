"""Gradio app for pass_6 — hand-built Fourier modular-addition head.

Demo tab:
  - headline grouped bar chart: mean alignment (addition vs difference vs random
    strawman vs analytic baseline);
  - per-frequency alignment & phase line plots;
  - interactive q(a).k(b') panel showing the score is a pure function of (a+b).
Benchmark tab: shared benchmark_panel across all attempts.

All charts read the latest run's artifacts.json — no fabricated numbers.
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

THIS_DIR = Path(__file__).parent
GOAL_DIR = THIS_DIR.parent
RESULTS_BASE = THIS_DIR / "results"
P = 97
D_HEAD = 128
N_FREQ = P // 2


# ----------------------------- artifact loading ----------------------------- #
def _run_dirs():
    if not RESULTS_BASE.exists():
        return []
    return sorted([d for d in RESULTS_BASE.iterdir()
                   if d.is_dir() and (d / "artifacts.json").exists()])


def _load(run_name=None):
    runs = _run_dirs()
    if not runs:
        return None
    chosen = runs[-1]
    if run_name:
        for d in runs:
            if d.name == run_name:
                chosen = d
                break
    return json.loads((chosen / "artifacts.json").read_text())


# ------------------------------- plots -------------------------------------- #
def _bar(art):
    base = art["random_baseline_alignment"]
    add = float(np.mean(art["add"]["alignment"]))
    diff = float(np.mean(art["diff"]["alignment"]))
    rand = float(np.mean(art["rand"]["alignment"]))
    labels = ["addition\nhead (a+b)", "difference\nhead (K=Q)", "random\nstrawman", "2/d_head\nbaseline"]
    vals = [add, diff, rand, base]
    colors = ["#2a7", "#7ac", "#c66", "#999"]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("mean fourier_alignment")
    ax.set_ylim(0, 1.12)
    ax.set_title("Headline alignment: real head vs strawman")
    fig.tight_layout()
    return fig


def _freq_lines(art):
    f = art["freq"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(f, art["add"]["alignment"], "-o", ms=3, label="addition", color="#2a7")
    ax1.plot(f, art["rand"]["alignment"], "-o", ms=3, label="random", color="#c66")
    ax1.axhline(art["random_baseline_alignment"], ls="--", color="#999", label="2/d_head")
    ax1.set_xlabel("frequency k"); ax1.set_ylabel("alignment"); ax1.set_ylim(0, 1.05)
    ax1.set_title("alignment per frequency"); ax1.legend(fontsize=8)
    ax2.plot(f, art["add"]["phase_error"], "-o", ms=3, label="addition (π/2, intrinsic)", color="#2a7")
    ax2.plot(f, art["diff"]["phase_error"], "-o", ms=3, label="difference K=Q (0)", color="#7ac")
    ax2.axhline(np.pi / 2, ls=":", color="#999")
    ax2.set_xlabel("frequency k"); ax2.set_ylabel("phase_error")
    ax2.set_ylim(-0.1, np.pi); ax2.set_title("phase_error per frequency"); ax2.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _qk_basis():
    x = np.arange(P)
    E = np.zeros((P, D_HEAD))
    for k in range(1, N_FREQ + 1):
        E[:, 2 * (k - 1)] = np.cos(2 * np.pi * k * x / P)
        E[:, 2 * (k - 1) + 1] = np.sin(2 * np.pi * k * x / P)
    s = np.ones(D_HEAD); s[1:2 * N_FREQ:2] = -1.0; s[2 * N_FREQ:] = 0.0
    return E, s


_E, _S = _qk_basis()


def _interactive(a):
    a = int(a) % P
    qa = _E[a]
    scores = (_E * _S) @ qa            # q(a) . k(b') for all b'
    bs = np.arange(P)
    sums = (a + bs) % P
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(bs, scores, color="#2a7")
    ax1.set_xlabel("b'"); ax1.set_ylabel("q(a)·k(b')")
    ax1.set_title(f"score vs b'   (a={a})")
    order = np.argsort(sums)
    ax2.plot(sums[order], scores[order], "-o", ms=3, color="#26a")
    ax2.set_xlabel("(a+b') mod p"); ax2.set_ylabel("q(a)·k(b')")
    ax2.set_title("same scores collapse onto (a+b)")
    fig.tight_layout()
    peak = int(sums[int(np.argmax(scores))])
    return fig, f"Score is a pure function of (a+b) mod p. Max score at a+b ≡ {peak} (mod {P})."


def _summary(art):
    if art is None:
        return "No run yet — execute main.py first."
    add = float(np.mean(art["add"]["alignment"]))
    addp = float(np.mean(art["add"]["phase_error"]))
    op = art.get("operating_range", [])
    lines = [
        f"**Scored addition head** — mean alignment **{add:.4f}**, "
        f"mean phase_error **{addp:.4f}** (= π/2, intrinsic for a pure-(a+b) head), "
        f"max alignment **{art['max_alignment']:.4f}** @ k={art['argmax_alignment_freq']}.",
        f"Baseline 2/d_head = {art['random_baseline_alignment']:.4f}; "
        f"independent-random strawman mean alignment = {np.mean(art['rand']['alignment']):.4f}.",
        "",
        "**Operating range (mean alignment / phase):**",
        "| p | mean_align | max_align | mean_phase |",
        "|---|-----------|-----------|------------|",
    ] + [f"| {r['modulus']} | {r['mean_alignment']:.4f} | {r['max_alignment']:.4f} | {r['mean_phase']:.4f} |"
         for r in op]
    return "\n".join(lines)


# ------------------------------- UI ----------------------------------------- #
with gr.Blocks() as demo:
    gr.Markdown("# attention_modular_add / pass_6 — hand-built Fourier addition head")
    gr.Markdown(
        "Single hand-set attention head. `Q(a)=E[a]`, `K(b)=E[b]·s` with `s` negating "
        "the sin channels ⇒ `q·k = Σ_k cos(k(a+b))`, a genuine **modular-addition** circuit. "
        "Headline alignment is maxed; phase_error sits at π/2 — which is *intrinsic* to a "
        "pure-(a+b) head under this metric (only a `K=Q` difference head reaches 0)."
    )

    with gr.Tab("Demo"):
        run_dd = gr.Dropdown(label="run", choices=[d.name for d in _run_dirs()],
                             value=(_run_dirs()[-1].name if _run_dirs() else None))
        summary_md = gr.Markdown()
        with gr.Row():
            bar_plot = gr.Plot(label="Headline alignment")
            freq_plot = gr.Plot(label="Per-frequency alignment & phase")
        gr.Markdown("### Interactive: the score is a pure function of (a+b)")
        with gr.Row():
            a_in = gr.Slider(0, P - 1, value=12, step=1, label="a")
        with gr.Row():
            qk_plot = gr.Plot(label="q(a)·k(b')")
        qk_note = gr.Markdown()

        def refresh(run_name):
            art = _load(run_name)
            if art is None:
                return None, None, "No run yet — execute main.py first."
            return _bar(art), _freq_lines(art), _summary(art)

        run_dd.change(refresh, inputs=run_dd, outputs=[bar_plot, freq_plot, summary_md])
        a_in.change(_interactive, inputs=a_in, outputs=[qk_plot, qk_note])
        demo.load(refresh, inputs=run_dd, outputs=[bar_plot, freq_plot, summary_md])
        demo.load(_interactive, inputs=a_in, outputs=[qk_plot, qk_note])

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
