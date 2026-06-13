"""Gradio app for attention_polyeval / pass_2.

Demo tab: the softmax-over-QK-square attention block evaluated live on CUDA,
showing (1) output tracking the x^2 parabola, (2) a degree-2 R^2 bar chart
comparing the mechanism vs. the linear-QK ablation vs. the linear baseline, and
(3) an operating-range curve across input scales.

Benchmark tab: cross-attempt leaderboard / history.
"""

import os
import json

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

DEVICE = "cuda"

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)
RESULTS_DIR = os.path.join(ATTEMPT_DIR, "results")

BETA = 1.0
B0 = 0.0


# ---------------------------------------------------------------------------
# The mechanism (mirrors main.py) — runs live on CUDA.
# ---------------------------------------------------------------------------
def calibrate(beta, b0, scale, mode, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-scale, scale, size=n)
    u = x * x
    ls = beta * x * x if mode == "quad" else beta * x
    p = 1.0 / (1.0 + np.exp(-(ls - b0)))
    A = np.stack([np.ones_like(p), p], axis=1)
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    return float(coef[1]), float(coef[0])  # alpha, gamma


def attention_forward(x_np, beta, b0, alpha, gamma, mode="quad"):
    x = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE)
    self_logit = beta * x * x if mode == "quad" else beta * x
    sink_logit = torch.full_like(self_logit, float(b0))
    logits = torch.stack([self_logit, sink_logit], dim=-1)
    p = torch.softmax(logits, dim=-1)[..., 0]
    out = gamma + alpha * p
    return out.detach().cpu().numpy().astype(np.float32)


def r2_vs(out, target):
    var = float(np.var(target))
    return 1.0 - float(np.mean((out - target) ** 2)) / var if var > 0 else 0.0


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------
def list_runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = [d for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))]
    return sorted(runs, reverse=True)


def _load(run, name):
    path = os.path.join(RESULTS_DIR, run, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_parabola(seed):
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-1.0, 1.0, size=(128, 64)).astype(np.float32)
    alpha, gamma = calibrate(BETA, B0, 1.0, "quad")
    out = attention_forward(x, BETA, B0, alpha, gamma, "quad")
    xf, of = x.flatten(), out.flatten()
    order = np.argsort(xf)
    xs = np.linspace(-1, 1, 200)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(xf[order], of[order], s=4, alpha=0.25, label="attention output")
    ax.plot(xs, xs ** 2, "r-", lw=2, label="target  x²")
    ax.set_xlabel("input  x")
    ax.set_ylabel("output")
    ax.set_title(f"Mechanism output vs x²   (R²={r2_vs(out, x**2):.4f})")
    ax.legend()
    fig.tight_layout()
    return fig


def fig_bars(run):
    abl = _load(run, "ablation.json") if run else None
    if abl is None:
        # recompute live
        rng = np.random.default_rng(42)
        x = rng.uniform(-1.0, 1.0, size=(128, 64)).astype(np.float32)
        t = x ** 2
        a, g = calibrate(BETA, B0, 1.0, "quad")
        rq = r2_vs(attention_forward(x, BETA, B0, a, g, "quad"), t)
        al, gl = calibrate(BETA, B0, 1.0, "linear")
        rl = r2_vs(attention_forward(x, BETA, B0, al, gl, "linear"), t)
        abl = {"mechanism_quad_qk_r2": rq, "ablation_linear_qk_r2": rl,
               "linear_baseline_r2": 0.0}
    labels = ["softmax(QK²)\n(mechanism)", "linear-QK\nablation", "linear\nbaseline"]
    vals = [abl["mechanism_quad_qk_r2"], abl["ablation_linear_qk_r2"],
            abl["linear_baseline_r2"]]
    colors = ["#2a9d8f", "#e76f51", "#bbbbbb"]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bars = ax.bar(labels, vals, color=colors)
    ax.set_ylabel("R²  on  x²  (degree 2)")
    ax.set_title("Causal ablation: the QK square does the work")
    ax.axhline(0, color="k", lw=0.8)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom" if v >= 0 else "top")
    fig.tight_layout()
    return fig


def fig_scale(run):
    sweep = _load(run, "scale_sweep.json") if run else None
    if sweep is None:
        scales = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
        sweep = []
        for s in scales:
            rng = np.random.default_rng(123)
            xs = rng.uniform(-s, s, size=(128, 64)).astype(np.float32)
            ts = xs ** 2
            bs = 1.0 / (s * s)
            a, g = calibrate(bs, B0, s, "quad")
            sweep.append({"scale": s,
                          "r2_adaptive_beta": r2_vs(attention_forward(xs, bs, B0, a, g), ts),
                          "r2_fixed_beta": np.nan})
    sc = [d["scale"] for d in sweep]
    ra = [d["r2_adaptive_beta"] for d in sweep]
    rf = [d.get("r2_fixed_beta", np.nan) for d in sweep]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.semilogx(sc, ra, "o-", color="#2a9d8f", label="scale-adaptive β")
    ax.semilogx(sc, rf, "s--", color="#e76f51", label="fixed β=1")
    ax.set_xlabel("input scale  (Uniform[-s, s])")
    ax.set_ylabel("R²  on  x²")
    ax.set_ylim(-0.1, 1.05)
    ax.set_title("Operating range  (≥ 3 orders of magnitude)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def update(run, seed):
    runs = list_runs()
    if run is None and runs:
        run = runs[0]
    return fig_parabola(seed), fig_bars(run), fig_scale(run)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# attention_polyeval — pass_2")
    gr.Markdown(
        "A single attention block evaluates **x²** elementwise. Each token attends "
        "to **itself** and one constant **sink** key; the self-score is the bilinear "
        "QK product **β·x²** (the squaring), softmax is the only nonlinearity, and "
        "an affine W_O reads it back out."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=list_runs(), value=(list_runs()[0] if list_runs() else None),
                                 label="results run")
            seed_sl = gr.Slider(0, 1000, value=42, step=1, label="random seed (parabola plot)")
        with gr.Row():
            p1 = gr.Plot(label="Output tracks x²")
            p2 = gr.Plot(label="Ablation R² (degree 2)")
        with gr.Row():
            p3 = gr.Plot(label="Operating range")

        run_dd.change(update, inputs=[run_dd, seed_sl], outputs=[p1, p2, p3])
        seed_sl.change(update, inputs=[run_dd, seed_sl], outputs=[p1, p2, p3])
        demo.load(update, inputs=[run_dd, seed_sl], outputs=[p1, p2, p3])

    with gr.Tab("Benchmark"):
        gr.Markdown("Leaderboard and metric history across all attempts for this goal.")
        try:
            benchmark_panel(GOAL_DIR)
        except Exception as e:  # keep boot-check alive even if panel errors
            gr.Markdown(f"_benchmark panel unavailable: {e}_")


if __name__ == "__main__":
    demo.launch()
