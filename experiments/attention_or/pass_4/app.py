"""attention_or / pass_4 — Demo + Benchmark tabs.

Demo panels:
  A) OR sharpness vs cos(q_A,q_B): the soft-max OR circuit (flat ~1.0) vs the
     plain-linear superposition STRAWMAN (sqrt((1+cos)/2), fails at low cos).
  B) Component ablations at the canonical anchor: full / no-softmax / no-gate /
     plain-linear, on BOTH sharpness and noise leakage — each broken part
     breaks a different desideratum (causal evidence for the circuit).
  C) beta sweep: the soft-max temperature is the OR knob (average -> hard max).
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_ROOT = Path(__file__).parent / "results"

C_OR = "#4e79a7"
C_LIN = "#e15759"
C_IDEAL = "#59a14f"


def get_run_choices():
    if not RESULTS_ROOT.exists():
        return []
    return [str(d) for d in sorted(RESULTS_ROOT.iterdir()) if d.is_dir()]


def _load_json(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _empty(msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_axis_off()
    return fig


def make_figs(run_dir_str):
    if not run_dir_str:
        e = _empty("No runs yet — run main.py first")
        return e, e, e
    run_dir = Path(run_dir_str)
    analysis = _load_json(run_dir / "analysis.json")
    bench = _load_json(run_dir / "benchmark.json")

    if analysis is None:
        e = _empty("analysis.json missing for this run")
        return e, e, e

    # ---- Panel A: sharpness vs cos -----------------------------------------
    sw = analysis["main_sweep"]
    cos = np.array([s["cos"] for s in sw])
    or_sh = np.array([s["or_sharpness"] for s in sw])
    lin_sh = np.array([s["plain_linear_sharpness"] for s in sw])

    figA, axA = plt.subplots(figsize=(7, 4.6))
    axA.axhline(1.0, color=C_IDEAL, ls=":", lw=2, label="ideal OR = 1.0")
    axA.plot(cos, or_sh, "o-", color=C_OR, lw=2.5,
             label="OR circuit (log-sum-exp soft-max)")
    axA.plot(cos, lin_sh, "s--", color=C_LIN, lw=2,
             label="plain-linear superposition (strawman)")
    axA.set_xlabel("cos(q_A, q_B)")
    axA.set_ylabel("OR sharpness  min(s_AB@A, s_AB@B) / max(s_A@A, s_B@B)")
    axA.set_title("OR sharpness across feature overlap")
    axA.set_ylim(0.0, 1.12)
    axA.legend(loc="lower right", fontsize=9)
    axA.grid(True, alpha=0.3)
    headline = analysis["ablation"]["full"]["sharpness"]
    axA.annotate(f"canonical (cos=0): {headline:.3f}", xy=(0.0, headline),
                 xytext=(0.18, 0.55), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=C_OR))
    figA.tight_layout()

    # ---- Panel B: ablation grouped bars ------------------------------------
    ab = analysis["ablation"]
    order = ["full", "no_softmax", "no_gate", "plain_linear"]
    labels = ["full\ncircuit", "no soft-max\n(beta->0)",
              "no gate\n(gamma=0)", "plain\nlinear"]
    sharp = [ab[k]["sharpness"] for k in order]
    leak = [ab[k]["noise_leakage"] for k in order]
    x = np.arange(len(order))
    w = 0.38

    figB, axB = plt.subplots(figsize=(7, 4.6))
    axB.bar(x - w / 2, sharp, w, color=C_OR, label="OR sharpness (higher=better)")
    axB.bar(x + w / 2, leak, w, color="#b07aa1",
            label="noise leakage (lower=better)")
    axB.axhline(1.0, color=C_IDEAL, ls=":", lw=1.5)
    axB.set_xticks(x)
    axB.set_xticklabels(labels, fontsize=9)
    axB.set_ylabel("metric value")
    axB.set_title("Component ablations at canonical anchor (cos=0)")
    axB.set_ylim(0.0, 1.25)
    axB.legend(fontsize=9)
    axB.grid(True, axis="y", alpha=0.3)
    for xi, v in zip(x - w / 2, sharp):
        axB.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, leak):
        axB.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    figB.tight_layout()

    # ---- Panel C: beta sweep -----------------------------------------------
    bc = analysis["beta_curve"]
    betas = np.array([b["beta"] for b in bc])
    bsh = np.array([b["sharpness"] for b in bc])

    figC, axC = plt.subplots(figsize=(7, 4.6))
    axC.axhline(1.0, color=C_IDEAL, ls=":", lw=2, label="ideal OR (max) = 1.0")
    axC.axhline(np.sqrt(0.5), color=C_LIN, ls="--", lw=1.5,
                label="plain-linear strawman = 0.707")
    axC.plot(betas, bsh, "o-", color=C_OR, lw=2.5, label="OR circuit")
    axC.set_xscale("log", base=2)
    axC.set_xlabel("soft-max temperature  beta  (log scale)")
    axC.set_ylabel("OR sharpness at cos=0")
    axC.set_title("The soft-max temperature is the OR knob: average -> hard max")
    axC.set_ylim(0.0, 1.12)
    axC.legend(loc="lower right", fontsize=9)
    axC.grid(True, alpha=0.3)
    figC.tight_layout()

    return figA, figB, figC


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_or — pass_4\n"
        "**Logical OR = log-sum-exp soft maximum inside attention.** One fixed "
        "circuit handles q_A, q_B and the superposition q_AB with no branching; "
        "the *max* is produced by the soft-max as the temperature beta grows."
    )

    with gr.Tabs():
        with gr.TabItem("Demo"):
            choices = get_run_choices()
            run_dd = gr.Dropdown(
                choices=choices,
                value=choices[-1] if choices else None,
                label="Run directory (defaults to latest)",
                interactive=True,
            )
            plotA = gr.Plot(label="A — OR sharpness vs feature overlap")
            with gr.Row():
                plotB = gr.Plot(label="B — component ablations")
                plotC = gr.Plot(label="C — beta is the OR knob")

            run_dd.change(make_figs, inputs=run_dd,
                          outputs=[plotA, plotB, plotC])
            demo.load(make_figs, inputs=run_dd,
                      outputs=[plotA, plotB, plotC])

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt benchmark history")
            benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
