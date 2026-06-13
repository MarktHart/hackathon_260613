"""Gradio app for attention_gcd / first_pass.

Demo tab  : the hand-built thermometer circuit — gcd is decoded from the
            residual stream at SEP by a counting readout, while the raw-[a,b]
            baseline cannot.
Benchmark : cross-attempt leaderboard via agentic.experiments.benchmark_panel.
"""

import json
import os
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)
RESULTS_DIR = os.path.join(ATTEMPT_DIR, "results")


def list_runs():
    runs = sorted(glob(os.path.join(RESULTS_DIR, "*")), reverse=True)
    return [os.path.basename(r) for r in runs if os.path.isdir(r)]


def _load(run_id):
    run = os.path.join(RESULTS_DIR, run_id)
    demo_npz = np.load(os.path.join(run, "demo.npz"))
    with open(os.path.join(run, "summary.json")) as f:
        summary = json.load(f)
    return demo_npz, summary


def make_figs(run_id):
    if not run_id:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No runs found — run main.py first", ha="center")
        ax.axis("off")
        return fig, fig, "No run selected."

    d, s = _load(run_id)

    # ---- Fig 1: predicted vs true gcd (headline decodability) -------------
    f1, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(d["gcd_true"], d["gcd_pred"], s=14, alpha=0.5,
               color="#1f77b4", label="residual probe")
    lim = [0, float(max(d["gcd_true"].max(), d["gcd_pred"].max())) + 1]
    ax.plot(lim, lim, "k--", lw=1, label="y = x (perfect)")
    ax.set_xlabel("true gcd(a, b)")
    ax.set_ylabel("linear probe prediction")
    ax.set_title(f"gcd decoded from residual @ SEP\n"
                 f"R² = {s['headline_resid_r2']:.4f}  ·  acc = {s['headline_resid_acc']:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    f1.tight_layout()

    # ---- Fig 2: thermometer staircase + metric bars -----------------------
    f2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))

    therm = d["therm_sample"]               # [n, MAX_N] rows sorted by gcd
    tg = d["therm_gcd"]
    im = axa.imshow(therm, aspect="auto", cmap="magma",
                    interpolation="nearest", origin="lower")
    axa.set_xlabel("threshold k  (t[k] = [gcd ≥ k])")
    axa.set_ylabel("samples (sorted by gcd ↑)")
    axa.set_title("Residual thermometer code  →  gcd = Σ_k t[k]")
    yt = np.linspace(0, len(tg) - 1, 6).astype(int)
    axa.set_yticks(yt)
    axa.set_yticklabels([f"{int(tg[i])}" for i in yt])
    f2.colorbar(im, ax=axa, fraction=0.046, pad=0.04)

    labels = ["resid R²", "resid acc", "attn corr"]
    method = [s["headline_resid_r2"], s["headline_resid_acc"], s["best_attn_corr"]]
    base = [max(0.0, s["baseline_resid_r2"]), s["baseline_resid_acc"],
            s["baseline_attn_corr"]]
    x = np.arange(len(labels))
    w = 0.38
    axb.bar(x - w / 2, method, w, label="thermometer circuit", color="#2ca02c")
    axb.bar(x + w / 2, base, w, label="raw-[a,b] baseline", color="#999999")
    axb.set_xticks(x)
    axb.set_xticklabels(labels)
    axb.set_ylim(0, 1.05)
    axb.set_title("Circuit vs. baseline")
    axb.legend(fontsize=8)
    for xi, mv in zip(x, method):
        axb.text(xi - w / 2, mv + 0.02, f"{mv:.2f}", ha="center", fontsize=8)
    f2.tight_layout()

    md = (
        f"### Run `{run_id}`\n"
        f"- **gcd decodability (R²)**: {s['headline_resid_r2']:.4f}  "
        f"(baseline {s['baseline_resid_r2']:.3f})\n"
        f"- **decode accuracy**: {s['headline_resid_acc']:.3f}  "
        f"(baseline {s['baseline_resid_acc']:.3f})\n"
        f"- **best attn SEP→operand corr**: {s['best_attn_corr']:.3f}  "
        f"(baseline {s['baseline_attn_corr']:.3f})\n\n"
        "gcd is read off the residual stream as a *count* of satisfied "
        "thermometer thresholds `t[k] = [gcd ≥ k]`, themselves a suffix-OR "
        "over the common-divisor indicators `[d | a and d | b]`. The raw "
        "operand baseline can't — gcd is non-linear in (a, b)."
    )
    return f1, f2, md


with gr.Blocks(title="attention_gcd · first_pass") as demo:
    gr.Markdown(
        "# attention_gcd — hand-built thermometer circuit\n"
        "Is `gcd(a, b)` linearly decodable from the residual at `SEP`? "
        "Here it is **by construction**: the residual carries a thermometer "
        "code of gcd built from common-divisor indicators, so a linear "
        "counting probe recovers gcd exactly."
    )
    runs = list_runs()
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=runs, value=(runs[0] if runs else None),
                label="run", scale=3,
            )
            refresh = gr.Button("↻ refresh", scale=1)
        info = gr.Markdown()
        with gr.Row():
            fig_scatter = gr.Plot(label="gcd decodability")
        with gr.Row():
            fig_detail = gr.Plot(label="thermometer code & baseline comparison")

        def _update(run_id):
            f1, f2, md = make_figs(run_id)
            return f1, f2, md

        def _refresh():
            rs = list_runs()
            val = rs[0] if rs else None
            f1, f2, md = make_figs(val)
            return gr.update(choices=rs, value=val), f1, f2, md

        run_dd.change(_update, inputs=run_dd,
                      outputs=[fig_scatter, fig_detail, info])
        refresh.click(_refresh, inputs=None,
                      outputs=[run_dd, fig_scatter, fig_detail, info])
        demo.load(_update, inputs=run_dd,
                  outputs=[fig_scatter, fig_detail, info])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
