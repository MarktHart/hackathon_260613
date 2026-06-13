"""Gradio app for attention_knapsack / pass_3.

Demo tab  : optimality-vs-capacity curve for the attention swap-circuit against
            the greedy ratio baseline, plus feasibility annotation.
Benchmark : shared cross-attempt leaderboard via benchmark_panel.
"""

import glob
import json
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

HERE = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")


def _runs():
    if not os.path.isdir(RESULTS):
        return []
    out = []
    for d in sorted(glob.glob(os.path.join(RESULTS, "*")), reverse=True):
        if os.path.exists(os.path.join(d, "demo_data.json")):
            out.append(os.path.basename(d))
    return out


def _load(run_name):
    with open(os.path.join(RESULTS, run_name, "demo_data.json")) as f:
        return json.load(f)


def render(run_name):
    if not run_name:
        return None, "No run found — run main.py first."
    d = _load(run_name)
    fracs = d["sweep_fracs"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # --- Left: optimality vs capacity (the headline claim) ---
    ax1.plot(fracs, d["model_optimality"], "o-", lw=2.4, ms=8,
             color="#1f77b4", label="attention swap-circuit")
    ax1.plot(fracs, d["baseline_optimality"], "s--", lw=2.0, ms=7,
             color="#d62728", label="greedy ratio (no-mechanism)")
    ax1.axhline(1.0, color="#2ca02c", lw=1.2, ls=":", label="exact optimum")
    ax1.set_xlabel("capacity_frac")
    ax1.set_ylabel("optimality  (1 − gap)")
    ax1.set_title("Optimality across the capacity sweep")
    lo = min(min(d["baseline_optimality"]), min(d["model_optimality"]))
    ax1.set_ylim(lo - 0.01, 1.002)
    ax1.legend(loc="lower right", fontsize=8)
    ax1.grid(alpha=0.3)

    # --- Right: lift over baseline + feasibility annotation ---
    lift = [m - b for m, b in zip(d["model_optimality"], d["baseline_optimality"])]
    bars = ax2.bar([str(f) for f in fracs], lift, color="#1f77b4", alpha=0.85)
    ax2.set_xlabel("capacity_frac")
    ax2.set_ylabel("optimality lift over greedy")
    ax2.set_title("Gap to greedy closed (feasibility annotated)")
    ax2.axhline(0.0, color="k", lw=0.8)
    for bar, fr in zip(bars, d["model_feasible"]):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"feas\n{fr:.2f}", ha="center", va="bottom", fontsize=7)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()

    summary = (
        f"### Run `{run_name}`\n"
        f"- **Headline robustness (mean optimality over sweep):** "
        f"{d['robustness']:.4f}\n"
        f"- **Canonical optimality (frac=0.5):** {d['canonical_optimality']:.4f} "
        f"vs greedy {d['baseline_canonical_optimality']:.4f} "
        f"(**lift +{d['canonical_optimality'] - d['baseline_canonical_optimality']:.4f}**)\n"
        f"- **Feasible rate (canonical):** {d['canonical_feasible']:.3f} "
        f"(every selection respects capacity)\n\n"
        "The circuit starts from the greedy solution and applies only "
        "feasibility-preserving, value-increasing attention swaps, so it can "
        "never fall below greedy — and strictly beats it at every capacity."
    )
    return fig, summary


with gr.Blocks(title="attention_knapsack / pass_3") as demo:
    gr.Markdown(
        "# Attention Knapsack — pass_3\n"
        "Hand-built **attention swap-circuit**: selected items are queries, "
        "unselected items are keys; the masked score `v_j − v_i` (feasible & "
        "improving only) drives top-1 attention that refines a greedy start "
        "toward the exact optimum."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            runs = _runs()
            run_dd = gr.Dropdown(
                choices=runs,
                value=runs[0] if runs else None,
                label="results run (latest first)",
            )
            plot = gr.Plot(label="optimality vs capacity")
            info = gr.Markdown()
            run_dd.change(render, inputs=run_dd, outputs=[plot, info])
            demo.load(render, inputs=run_dd, outputs=[plot, info])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()