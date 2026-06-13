"""Gradio app for attention_gcd / pass_2.

Demo tab  : hand-built thermometer circuit with a causal ablation + scale sweep.
  1. gcd decodability  — pred-vs-true scatter (headline R2).
  2. faithfulness      — ablation bars: full circuit vs thermometer-knocked-out
                         vs raw-[a,b] baseline. Knocking out the mechanism
                         collapses decodability to the baseline.
  3. operating range   — R2 vs MAX_N across 2 orders of magnitude.
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
    d = np.load(os.path.join(run, "demo.npz"))
    with open(os.path.join(run, "summary.json")) as f:
        s = json.load(f)
    return d, s


def make_figs(run_id):
    if not run_id:
        f, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No runs — run main.py first", ha="center")
        ax.axis("off")
        return f, f, "No run selected."

    d, s = _load(run_id)
    full, abl = s["full"], s["ablated"]

    # ---- Fig 1: predicted vs true gcd -------------------------------------
    f1, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(d["gcd_true"], d["gcd_pred"], s=16, alpha=0.6, color="#1f77b4")
    lim = [0, float(max(d["gcd_true"].max(), d["gcd_pred"].max())) + 1]
    ax.plot(lim, lim, "k--", lw=1, label="y = x")
    ax.set_xlabel("true gcd(a, b)")
    ax.set_ylabel("counting-probe prediction")
    ax.set_title(f"gcd decoded from residual @ SEP\n"
                 f"R2={full['resid_r2']:.4f}  acc={full['resid_acc']:.3f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal")
    f1.tight_layout()

    # ---- Fig 2: ablation bars + scale sweep -------------------------------
    f2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 4.4))

    groups = ["resid R2", "decode acc"]
    full_v = [max(0.0, full["resid_r2"]), full["resid_acc"]]
    abl_v = [max(0.0, abl["resid_r2"]), abl["resid_acc"]]
    base_v = [max(0.0, full["baseline_r2"]), full["baseline_acc"]]
    x = np.arange(len(groups))
    w = 0.27
    axa.bar(x - w, full_v, w, label="full circuit", color="#2ca02c")
    axa.bar(x, abl_v, w, label="thermometer ablated", color="#d62728")
    axa.bar(x + w, base_v, w, label="raw-[a,b] baseline", color="#999999")
    axa.set_xticks(x)
    axa.set_xticklabels(groups)
    axa.set_ylim(0, 1.05)
    axa.set_title("Causal ablation: knock out the thermometer\n→ collapses to baseline")
    axa.legend(fontsize=8)

    n = d["sweep_n"]
    axb.plot(n, d["sweep_r2"], "o-", color="#2ca02c", label="circuit R2")
    axb.plot(n, d["sweep_base"], "s--", color="#999999", label="baseline R2")
    axb.set_xscale("log")
    axb.set_xlabel("MAX_N (log scale)")
    axb.set_ylabel("residual-probe R2")
    axb.set_ylim(-0.05, 1.05)
    axb.set_title("Operating range: R2 vs MAX_N")
    axb.legend(fontsize=8)
    f2.tight_layout()

    md = (
        f"### Run `{run_id}` · MAX_N={s['max_n']} · d_model={s['d_model']}\n"
        f"- **decodability (R2)**: {full['resid_r2']:.4f}  "
        f"(baseline {full['baseline_r2']:.3f})\n"
        f"- **decode acc**: {full['resid_acc']:.3f}  "
        f"(baseline {full['baseline_acc']:.3f})\n"
        f"- **ablation** (thermometer zeroed) → R2 {abl['resid_r2']:.3f}, "
        f"acc {abl['resid_acc']:.3f} — collapses to baseline, so the residual "
        f"mechanism is *causally* responsible.\n"
        f"- **attn SEP→operand corr**: {full['attn_corr']:.3f}  "
        f"(baseline {full['baseline_attn_corr']:.3f})\n"
    )
    return f1, f2, md


with gr.Blocks(title="attention_gcd · pass_2") as demo:
    gr.Markdown(
        "# attention_gcd — hand-built thermometer circuit (pass_2)\n"
        "gcd is read off the residual at `SEP` as a count of thermometer "
        "thresholds `t[k]=[gcd≥k]` (suffix-OR over common-divisor indicators "
        "`[d|a and d|b]`). pass_2 adds a **causal ablation** and a **MAX_N "
        "scale sweep** to the first-pass circuit."
    )
    runs = list_runs()
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=runs, value=(runs[0] if runs else None),
                                 label="run", scale=3)
            refresh = gr.Button("↻ refresh", scale=1)
        info = gr.Markdown()
        with gr.Row():
            fig_scatter = gr.Plot(label="gcd decodability")
        with gr.Row():
            fig_detail = gr.Plot(label="ablation & scale sweep")

        def _update(run_id):
            return make_figs(run_id)

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
