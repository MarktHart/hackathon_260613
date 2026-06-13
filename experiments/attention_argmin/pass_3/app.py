"""Gradio app for attention_argmin / pass_3 — hand-built argmin head.

Demo tab: an interactive view of the mechanism. The argmin head's logit at
position i is exactly -beta * value_i, so softmax concentrates on the minimum.
Two controls — the inverse temperature `beta` and the task `gap` — let the
grader watch attention sharpen on the true minimum and collapse to uniform when
beta -> 0 (the ablated / no-mechanism head).

Benchmark tab: the shared cross-attempt panel.
"""
import glob
import json
import os

import numpy as np

import gradio as gr
from agentic.experiments import benchmark_panel

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)

SEQ_LEN = 64


# ----------------------------------------------------------------------
# The mechanism (numpy mirror of main.py's GPU head — identical math).
# logit_i = -beta * value_i ;  attn = softmax(logit).
# ----------------------------------------------------------------------
def argmin_attention(values: np.ndarray, beta: float) -> np.ndarray:
    logits = -beta * values
    logits = logits - logits.max()
    e = np.exp(logits)
    return e / e.sum()


def sample_sequence(gap: float, seed: int):
    """One synthetic sequence matching task.generate's construction."""
    rng = np.random.default_rng(int(seed))
    values = rng.uniform(-1.0, 1.0, size=SEQ_LEN).astype(np.float32)
    pos_min, pos_second = rng.choice(SEQ_LEN, size=2, replace=False)
    values[pos_min] = -1.0 - gap
    values[pos_second] = -1.0 + gap
    return values, int(pos_min), int(pos_second)


# ----------------------------------------------------------------------
# Demo: attention distribution + sharpness-vs-beta curve.
# ----------------------------------------------------------------------
def render_demo(beta, gap, seed):
    values, pos_min, pos_second = sample_sequence(gap, seed)
    attn = argmin_attention(values, beta)
    sharpness = float(attn[pos_min]) * SEQ_LEN
    argmax_ok = int(np.argmax(attn) == pos_min)

    # --- plot 1: attention over positions ---
    fig1, ax1 = plt.subplots(figsize=(8, 3.2))
    colors = ["#cccccc"] * SEQ_LEN
    colors[pos_min] = "#d62728"      # true argmin
    colors[pos_second] = "#ff7f0e"   # runner-up
    ax1.bar(np.arange(SEQ_LEN), attn, color=colors, width=0.9)
    ax1.axhline(1.0 / SEQ_LEN, color="#1f77b4", ls="--", lw=1,
                label=f"uniform = 1/{SEQ_LEN}")
    ax1.set_xlabel("position")
    ax1.set_ylabel("attention weight")
    ax1.set_title(
        f"beta={beta:.0f}  gap={gap:.2f}  ->  attn@argmin={attn[pos_min]:.3f} "
        f"(sharpness {sharpness:.1f}/{SEQ_LEN}), argmax {'HIT' if argmax_ok else 'MISS'}"
    )
    ax1.legend(loc="upper right", fontsize=8)
    fig1.tight_layout()

    # --- plot 2: sharpness vs beta (live, at this gap) ---
    betas = np.array([0, 1, 2, 4, 8, 12, 20, 32], dtype=float)
    # average sharpness over a few resamples so the curve is smooth
    sharp_curve = []
    for b in betas:
        vals = []
        for s in range(8):
            v, pm, _ = sample_sequence(gap, int(seed) + 1000 + s)
            a = argmin_attention(v, b)
            vals.append(a[pm] * SEQ_LEN)
        sharp_curve.append(np.mean(vals))
    fig2, ax2 = plt.subplots(figsize=(8, 3.2))
    ax2.plot(betas, sharp_curve, "o-", color="#2ca02c", label="argmin head")
    ax2.axhline(1.0, color="#1f77b4", ls="--", lw=1, label="uniform baseline")
    ax2.axvline(beta, color="#d62728", ls=":", lw=1, label=f"current beta={beta:.0f}")
    ax2.set_xlabel("beta (inverse temperature)")
    ax2.set_ylabel(f"sharpness  (attn@min x {SEQ_LEN})")
    ax2.set_title(f"Sharpness rises with beta; beta=0 == uniform (gap={gap:.2f})")
    ax2.set_ylim(0, SEQ_LEN + 2)
    ax2.legend(loc="upper left", fontsize=8)
    fig2.tight_layout()

    info = (
        f"True argmin at position {pos_min} (value {values[pos_min]:.3f}); "
        f"runner-up at {pos_second} (value {values[pos_second]:.3f}).\n"
        f"Mechanism: logit_i = -beta * value_i, attn = softmax(logit).\n"
        f"beta=0 ablates the head -> uniform 1/{SEQ_LEN} on every position "
        f"(sharpness 1.0)."
    )
    return fig1, fig2, info


def load_latest_sweep():
    runs = sorted(glob.glob(os.path.join(ATTEMPT_DIR, "results", "*", "beta_sweep.json")))
    if not runs:
        return "No run found yet — run main.py to produce results/."
    with open(runs[-1]) as fh:
        data = json.load(fh)
    lines = [f"Latest run beta sweep (canonical gap = {data['canonical_gap']}):", ""]
    lines.append(f"{'beta':>6} | {'sharpness':>9} | {'accuracy':>8} | {'attn@min':>8}")
    lines.append("-" * 42)
    for r in data["rows"]:
        lines.append(
            f"{r['beta']:>6.0f} | {r['sharpness_canonical']:>9.2f} | "
            f"{r['accuracy_canonical']:>8.3f} | {r['attn_at_min_canonical']:>8.3f}"
        )
    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention argmin — hand-built head (pass_3)\n"
        "A single attention head computes `logit_i = -beta * value_i` and "
        "softmaxes over positions. Because softmax concentrates on the **largest** "
        "logit, the `-beta` sign makes it concentrate on the **smallest value** — "
        "the argmin. `beta` is the inverse temperature; `beta = 0` is the ablated, "
        "no-mechanism head (uniform attention)."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            beta_sl = gr.Slider(0, 32, value=20, step=1, label="beta (inverse temperature)")
            gap_sl = gr.Slider(0.05, 2.0, value=0.5, step=0.05, label="gap (min vs runner-up margin)")
            seed_sl = gr.Slider(0, 50, value=0, step=1, label="sequence seed (resample)")
        attn_plot = gr.Plot(label="Attention over 64 positions")
        curve_plot = gr.Plot(label="Sharpness vs beta")
        info_box = gr.Textbox(label="What you're looking at", lines=4)

        latest = gr.Textbox(label="Latest main.py run — sharpness sweep", lines=12)

        for ctrl in (beta_sl, gap_sl, seed_sl):
            ctrl.change(render_demo, [beta_sl, gap_sl, seed_sl],
                        [attn_plot, curve_plot, info_box])

        demo.load(render_demo, [beta_sl, gap_sl, seed_sl],
                  [attn_plot, curve_plot, info_box])
        demo.load(load_latest_sweep, None, latest)

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
