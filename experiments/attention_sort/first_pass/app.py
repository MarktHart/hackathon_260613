"""Gradio app for attention_sort / first_pass.

Demo tab      : the sorting head in action — an attention heatmap where the
                bright diagonal-in-rank-space *is* the sort, with the ground
                truth argsort overlaid; plus accuracy-vs-length (the robustness
                story) and accuracy-vs-temperature (the mechanism is a sharp
                comparison limit).
Benchmark tab : cross-attempt leaderboard / history via benchmark_panel.
"""
import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

from agentic.experiments import load_task, benchmark_panel

GOAL_DIR = Path(__file__).resolve().parent.parent          # experiments/attention_sort
ATTEMPT_DIR = Path(__file__).resolve().parent
RESULTS = ATTEMPT_DIR / "results"

task = load_task(__file__)


# ----------------------------------------------------------------------------
# The sorting head (pure NumPy mirror of main.sorting_head_logits — the app
# does not need a GPU; the benchmark already ran on one).
# ----------------------------------------------------------------------------
def sorting_attn(values: np.ndarray, tau: float, beta: float) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)               # [N, L]
    diff = v[:, :, None] - v[:, None, :]
    soft_rank = 1.0 / (1.0 + np.exp(-np.clip(tau * diff, -60.0, 60.0)))
    soft_rank = soft_rank.sum(axis=2) - 0.5                 # [N, L]
    idx = np.arange(v.shape[1])
    logits = -beta * (soft_rank[:, None, :] - idx[None, :, None]) ** 2
    z = logits - logits.max(axis=2, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=2, keepdims=True)                 # [N, L, L]


# ----------------------------------------------------------------------------
# Run discovery
# ----------------------------------------------------------------------------
def list_runs():
    if not RESULTS.exists():
        return []
    return sorted([p.name for p in RESULTS.iterdir() if p.is_dir()], reverse=True)


def _load_json(run, name):
    p = RESULTS / run / name
    if p.exists():
        return json.loads(p.read_text())
    return None


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------
def plot_heatmap(run, length, tau, beta):
    L = int(length)
    samples = _load_json(run, "samples.json") if run else None
    if samples and str(L) in samples and abs(tau - 1.0e4) < 1e-6:
        rec = samples[str(L)]
        vals = np.array(rec["values"])
        attn = np.array(rec["attn"])
        target = rec["target_key"]
    else:
        # recompute live (e.g. when the user moves the temperature slider)
        batch = task.generate(seed=task.EVAL_SEED)
        vals = batch.sequences[L][0]
        attn = sorting_attn(vals[None, :], tau, beta)[0]
        target = np.argsort(vals).tolist()

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(attn, cmap="magma", vmin=0, vmax=1, aspect="equal")
    # overlay ground-truth target key for each output slot
    ax.scatter(target, np.arange(L), s=70, facecolors="none",
               edgecolors="cyan", linewidths=1.8, label="argsort target")
    ax.set_xlabel("input position (key)")
    ax.set_ylabel("output slot i  (wants i-th smallest)")
    ax.set_title(f"L={L}  attention   (cyan ○ = correct key)")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention mass")
    fig.tight_layout()
    return fig


def plot_accuracy_vs_length(run):
    bench = _load_json(run, "benchmark.json") if run else None
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if bench and "metrics" in bench:
        m = bench["metrics"]
        lens, accs = [], []
        for L in (4, 8, 16, 32):
            k = f"sort_accuracy_len_{L}"
            if k in m:
                lens.append(L)
                accs.append(m[k])
        ax.plot(lens, accs, "o-", color="#d6336c", lw=2, label="sorting head")
        ax.plot(lens, [1.0 / L for L in lens], "s--", color="gray",
                label="uniform (1/L)")
        rob = m.get("sort_robustness", float("nan"))
        ax.set_title(f"sort_robustness = {rob:.3f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("sequence length L")
    ax.set_ylabel("argmax-key accuracy")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_tau_sweep(run):
    sweep = _load_json(run, "tau_sweep.json") if run else None
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    if sweep:
        taus = [r["tau"] for r in sweep]
        accs = [r["accuracy"] for r in sweep]
        ax.plot(taus, accs, "o-", color="#1c7ed6", lw=2)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("comparison sharpness  τ  (log)")
    ax.set_ylabel("accuracy @ L=8")
    ax.set_title("counting head: soft → sharp comparison")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def refresh(run, length, tau, beta):
    return (plot_heatmap(run, length, tau, beta),
            plot_accuracy_vs_length(run),
            plot_tau_sweep(run))


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
runs = list_runs()
default_run = runs[0] if runs else None

with gr.Blocks(title="attention_sort / first_pass") as demo:
    gr.Markdown(
        "# attention_sort — rank-routing sorting head\n"
        "A hand-built attention head sorts by (1) **counting** how many values "
        "lie below each token to get its rank, then (2) **routing** output slot "
        "`i` to the token of rank `i`. Cyan circles mark the `argsort` target — "
        "bright cells should sit on them."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(choices=runs, value=default_run,
                                     label="run")
                length_dd = gr.Dropdown(choices=["4", "8", "16", "32"],
                                        value="8", label="sequence length")
            with gr.Row():
                tau_sl = gr.Slider(0.5, 1.0e4, value=1.0e4, label="τ (comparison sharpness)")
                beta_sl = gr.Slider(1.0, 60.0, value=30.0, label="β (routing sharpness)")
            with gr.Row():
                heat = gr.Plot(label="attention heatmap")
                with gr.Column():
                    acc_plot = gr.Plot(label="accuracy vs length (robustness)")
                    tau_plot = gr.Plot(label="accuracy vs τ")

            inputs = [run_dd, length_dd, tau_sl, beta_sl]
            outputs = [heat, acc_plot, tau_plot]
            run_dd.change(refresh, inputs=inputs, outputs=outputs)
            length_dd.change(refresh, inputs=inputs, outputs=outputs)
            tau_sl.change(refresh, inputs=inputs, outputs=outputs)
            beta_sl.change(refresh, inputs=inputs, outputs=outputs)
            demo.load(refresh, inputs=inputs, outputs=outputs)

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
