"""Gradio app for attention_longest_run / first_pass.

Demo tab: inspect a single sample's attention-weight vector, the 0.5
threshold, the implanted true run, and the run our circuit detects; plus
summary bars (denoised vs raw-threshold vs predict-the-mean baseline).

Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""
import json
import os
from glob import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)
RESULTS_DIR = os.path.join(ATTEMPT_DIR, "results")


# ---------------------------------------------------------------- run loading
def list_runs():
    runs = sorted(glob(os.path.join(RESULTS_DIR, "*")), reverse=True)
    return [os.path.basename(r) for r in runs if os.path.isdir(r)]


def _load_json(run_id, name):
    p = os.path.join(RESULTS_DIR, run_id, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


# ------------------------------------------------------------- numpy circuit
def _close_np(b):
    """Morphological closing (dilate then erode), kernel 3, border-clamped."""
    def dil(x):
        return np.maximum.reduce([np.concatenate([[x[0]], x[:-1]]), x,
                                  np.concatenate([x[1:], [x[-1]]])])
    def ero(x):
        return np.minimum.reduce([np.concatenate([[x[0]], x[:-1]]), x,
                                  np.concatenate([x[1:], [x[-1]]])])
    return ero(dil(b))


def _longest_run_np(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v > 0 else 0
        best = max(best, run)
    return best


# ----------------------------------------------------------------- demo plots
def sample_plot(run_id, L, head):
    samples = _load_json(run_id, "samples.json")
    if samples is None:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "no samples.json for this run", ha="center")
        return fig
    L = int(L); head = int(head)
    rec = next((s for s in samples["samples"] if s["run_length"] == L),
               samples["samples"][0])
    thr = samples["threshold"]
    w = np.asarray(rec["weights"][head], dtype=np.float32)
    start, true_L = rec["start"], rec["run_length"]

    mask_raw = (w > thr).astype(np.float32)
    mask_den = _close_np(mask_raw)
    pred_raw = _longest_run_np(mask_raw)
    pred_den = _longest_run_np(mask_den)
    d = rec["difficulty_per_head"][head]

    fig, ax = plt.subplots(figsize=(9, 3.2))
    x = np.arange(len(w))
    ax.bar(x, w, width=0.9, color=np.where(mask_den > 0, "#1f77b4", "#cccccc"))
    ax.axhline(thr, color="red", lw=1.2, ls="--", label=f"threshold={thr}")
    ax.axvspan(start - 0.5, start + true_L - 0.5, color="orange", alpha=0.25,
               label=f"true run (L={true_L})")
    ax.set_xlabel("sequence position")
    ax.set_ylabel("attention weight")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"head {head} (difficulty d={d})  |  true L={true_L}   "
                 f"denoised pred={pred_den}   raw pred={pred_raw}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def summary_plot(run_id):
    comp = _load_json(run_id, "comparison.json")
    fig, ax = plt.subplots(figsize=(7, 4))
    if comp is None:
        ax.text(0.5, 0.5, "no comparison.json", ha="center")
        return fig
    diffs = comp["difficulties"]
    den = [comp["mae_denoised_by_d"][str(d)] if str(d) in comp["mae_denoised_by_d"]
           else comp["mae_denoised_by_d"][d] for d in diffs]
    raw = [comp["mae_raw_by_d"][str(d)] if str(d) in comp["mae_raw_by_d"]
           else comp["mae_raw_by_d"][d] for d in diffs]
    xs = np.arange(len(diffs))
    ax.bar(xs - 0.2, raw, width=0.4, label="raw threshold", color="#d62728")
    ax.bar(xs + 0.2, den, width=0.4, label="denoised (closing)", color="#1f77b4")
    ax.axhline(comp["baseline_mae"], color="gray", ls="--",
               label=f"predict-mean baseline ({comp['baseline_mae']:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d={d}" for d in diffs])
    ax.set_ylabel("MAE  (lower = better)")
    ax.set_title("Longest-run MAE by head difficulty")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def refresh(run_id, L, head):
    return sample_plot(run_id, L, head), summary_plot(run_id)


# ----------------------------------------------------------------------- app
_runs = list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks() as demo:
    gr.Markdown("# attention_longest_run — first_pass\n"
                "Hand-built circuit: threshold attention at 0.5, morphologically "
                "close single-position noise gaps, then take the longest "
                "contiguous run.")
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(_runs, value=_default, label="run")
                L_dd = gr.Dropdown([1, 3, 5, 8, 12, 16], value=8,
                                   label="true run length L")
                head_dd = gr.Dropdown(list(range(8)), value=1,
                                      label="head (d=0.3,0.5,0.7,0.9 cycling)")
            sample_fig = gr.Plot(label="attention weights & detected run")
            summary_fig = gr.Plot(label="MAE by difficulty")

            run_dd.change(refresh, [run_dd, L_dd, head_dd], [sample_fig, summary_fig])
            L_dd.change(sample_plot, [run_dd, L_dd, head_dd], sample_fig)
            head_dd.change(sample_plot, [run_dd, L_dd, head_dd], sample_fig)
            demo.load(refresh, [run_dd, L_dd, head_dd], [sample_fig, summary_fig])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
