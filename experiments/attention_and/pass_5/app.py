"""
Gradio app for attention_and / pass_5 ("magnitude AND" head).

Demo tab
  1. Robustness curve: AND-sharpness vs cos(q_A, q_B) for the magnitude-gate
     head against the linear baseline and three ablations. The headline claim
     ("flat near 1.0 across superposition") is the flat top line; the failures
     are the lines that dive as cos -> 1.
  2. Separation view: at a chosen cosine, the gate input S coloured by the
     ground-truth AND label, with the decision threshold drawn in. Shows the
     mechanism literally splitting "both" from "one/none".
Benchmark tab
  Cross-attempt leaderboard via agentic.experiments.benchmark_panel.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

_STYLE = {
    "and_head": ("magnitude AND head (ours)", "#1b7837", 2.6, "-", "o"),
    "linear_baseline": ("linear baseline (sum, no gate)", "#999999", 1.8, "--", "s"),
    "no_threshold": ("ablation: no gate (linear S)", "#762a83", 1.6, ":", "^"),
    "no_cosnorm": ("ablation: no (1+cos) correction", "#d95f02", 1.6, "-.", "v"),
    "product_gate": ("ablation: per-feature product (pass_4)", "#c51b7d", 1.6, "--", "D"),
}


def _run_dirs():
    if not RESULTS.exists():
        return []
    return sorted((p for p in RESULTS.iterdir() if p.is_dir()), reverse=True)


def _run_choices():
    return [p.name for p in _run_dirs()]


def _resolve(run_name):
    dirs = _run_dirs()
    if not dirs:
        return None
    if run_name:
        for p in dirs:
            if p.name == run_name:
                return p
    return dirs[0]


def robustness_plot(run_name):
    run = _resolve(run_name)
    fig = plt.figure(figsize=(7.2, 4.6), dpi=110)
    ax = fig.add_subplot(111)
    if run is None or not (run / "ablations.json").exists():
        ax.text(0.5, 0.5, "No run found — execute main.py first.",
                ha="center", va="center")
        ax.axis("off")
        return fig
    data = json.loads((run / "ablations.json").read_text())
    cos = data["cos_sweep"]
    for key, (label, color, lw, ls, marker) in _STYLE.items():
        if key not in data:
            continue
        ax.plot(cos, data[key], label=label, color=color, lw=lw,
                linestyle=ls, marker=marker, ms=5)
    ax.set_xlabel("cos(q_A, q_B)  —  feature overlap / superposition")
    ax.set_ylabel("AND sharpness  (0 = none, 1 = perfect)")
    ax.set_title("Does the AND boundary survive superposition?")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.0, color="black", lw=0.6)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def _summary_md(run_name):
    run = _resolve(run_name)
    if run is None or not (run / "ablations.json").exists():
        return "_No run found._"
    d = json.loads((run / "ablations.json").read_text())
    ours, base = d["and_head"], d["linear_baseline"]
    robust = ours[-1] / max(ours[0], 1e-9)
    prod = d.get("product_gate", [0])
    return (
        f"**Headline (this run):**  superposition_robustness = "
        f"`{min(max(robust,0),1):.3f}`  ·  sharpness@cos0 = `{ours[0]:.3f}`  ·  "
        f"lift over linear baseline = `{ours[0]-base[0]:+.3f}`\n\n"
        f"At full overlap (cos=1.0): magnitude head = `{ours[-1]:.3f}` vs "
        f"per-feature product gate = `{prod[-1]:.3f}` — the product gate's Gram "
        f"matrix is singular there, so it collapses while the magnitude gate holds."
    )


def separation_plot(run_name, cos_choice):
    run = _resolve(run_name)
    fig = plt.figure(figsize=(7.2, 4.0), dpi=110)
    ax = fig.add_subplot(111)
    key = f"cos_{float(cos_choice):.1f}"
    if run is None or not (run / "separation.json").exists():
        ax.text(0.5, 0.5, "No separation snapshot — run main.py.",
                ha="center", va="center")
        ax.axis("off")
        return fig
    snaps = json.loads((run / "separation.json").read_text())
    if key not in snaps:
        key = next(iter(snaps))
    snap = snaps[key]
    S = np.array(snap["S"])
    lab = np.array(snap["label_and"]).astype(bool)
    thr = snap["threshold"]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.18, 0.18, size=S.shape)
    ax.scatter(S[~lab], jit[~lab], s=22, color="#999999", alpha=0.7,
               label="NOT (A and B)")
    ax.scatter(S[lab], jit[lab], s=42, color="#1b7837", alpha=0.9,
               edgecolor="black", linewidth=0.4, label="A and B (target)")
    ax.axvline(thr, color="#d62728", lw=2.0, label=f"gate threshold = {thr}")
    ax.set_yticks([])
    ax.set_xlabel("gate input  S = (a+b) / (2(1+cos))   ≈  number of features present")
    ax.set_title(f"Magnitude separation at cos(q_A,q_B) = {float(cos_choice):.1f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_and · pass_5 — magnitude AND head\n"
        "A single attention head implements logical AND by **thresholding the "
        "total feature magnitude** instead of multiplying per-feature estimates. "
        "Because count-of-features stays observable even when q_A and q_B merge, "
        "the AND boundary survives full superposition (cos → 1) where a "
        "per-feature product gate collapses."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(
                    choices=_run_choices(),
                    value=(_run_choices()[0] if _run_choices() else None),
                    label="result run", interactive=True, scale=3,
                )
            summary = gr.Markdown(_summary_md(None))
            robust_fig = gr.Plot(label="AND sharpness vs superposition")
            gr.Markdown(
                "### Where does the AND signal actually live?\n"
                "Pick a cosine and watch the gate input split the target "
                "(`A and B`) from everything else around the red threshold."
            )
            cos_pick = gr.Radio(
                choices=["0.0", "1.0"], value="0.0",
                label="cosine snapshot", interactive=True,
            )
            sep_fig = gr.Plot(label="magnitude separation")

            def _refresh(run_name, cos_choice):
                return (robustness_plot(run_name), _summary_md(run_name),
                        separation_plot(run_name, cos_choice))

            run_dd.change(_refresh, inputs=[run_dd, cos_pick],
                          outputs=[robust_fig, summary, sep_fig])
            cos_pick.change(separation_plot, inputs=[run_dd, cos_pick],
                            outputs=sep_fig)
            demo.load(_refresh, inputs=[run_dd, cos_pick],
                      outputs=[robust_fig, summary, sep_fig])

        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
