"""Gradio app for attention_and / pass_6 — magnitude AND head."""
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
    "linear_baseline": ("linear baseline (a+b, no gate)", "#999999", 1.8, "--", "s"),
    "no_gate": ("ablation: drop the gate", "#762a83", 1.6, ":", "^"),
    "product_gate": ("ablation: per-feature product", "#c51b7d", 1.6, "-.", "D"),
}


def _run_dirs():
    if not RESULTS.exists():
        return []
    return sorted((p for p in RESULTS.iterdir() if p.is_dir()), reverse=True)


def _run_choices():
    return [p.name for p in _run_dirs()]


def _resolve(name):
    dirs = _run_dirs()
    if not dirs:
        return None
    for p in dirs:
        if p.name == name:
            return p
    return dirs[0]


def robustness_plot(name):
    run = _resolve(name)
    fig = plt.figure(figsize=(7.2, 4.6), dpi=110)
    ax = fig.add_subplot(111)
    if run is None or not (run / "ablations.json").exists():
        ax.text(0.5, 0.5, "No run — execute main.py first.", ha="center", va="center")
        ax.axis("off")
        return fig
    d = json.loads((run / "ablations.json").read_text())
    cos = d["cos_sweep"]
    for k, (lab, col, lw, ls, mk) in _STYLE.items():
        if k in d:
            ax.plot(cos, d[k], label=lab, color=col, lw=lw, linestyle=ls, marker=mk, ms=5)
    ax.set_xlabel("cos(q_A, q_B)  —  feature overlap / superposition")
    ax.set_ylabel("AND sharpness  (0=none, 1=perfect)")
    ax.set_title("Does the AND boundary survive superposition?")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def _summary(name):
    run = _resolve(name)
    if run is None or not (run / "ablations.json").exists():
        return "_No run found._"
    d = json.loads((run / "ablations.json").read_text())
    o, b = d["and_head"], d["linear_baseline"]
    robust = min(max(o[-1] / max(o[0], 1e-9), 0), 1)
    prod = d.get("product_gate", [0])
    return (
        f"**superposition_robustness = `{robust:.3f}`**  ·  sharpness@cos0 = "
        f"`{o[0]:.3f}`  ·  lift over baseline = `{o[0]-b[0]:+.3f}`\n\n"
        f"At cos=1.0: magnitude head `{o[-1]:.3f}` vs per-feature product gate "
        f"`{prod[-1]:.3f}` — the product gate's Gram matrix is singular there, so "
        f"it collapses while the magnitude gate holds."
    )


def separation_plot(name, cos_choice):
    run = _resolve(name)
    fig = plt.figure(figsize=(7.2, 4.0), dpi=110)
    ax = fig.add_subplot(111)
    if run is None or not (run / "separation.json").exists():
        ax.text(0.5, 0.5, "No snapshot — run main.py.", ha="center", va="center")
        ax.axis("off")
        return fig
    snaps = json.loads((run / "separation.json").read_text())
    key = f"cos_{float(cos_choice):.1f}"
    if key not in snaps:
        key = next(iter(snaps))
    snap = snaps[key]
    S = np.array(snap["S"])
    lab = np.array(snap["label_and"]).astype(bool)
    thr = snap["threshold"]
    rng = np.random.default_rng(0)
    jit = rng.uniform(-0.18, 0.18, size=S.shape)
    ax.scatter(S[~lab], jit[~lab], s=22, color="#999999", alpha=0.7, label="NOT (A and B)")
    ax.scatter(S[lab], jit[lab], s=42, color="#1b7837", alpha=0.9,
               edgecolor="black", linewidth=0.4, label="A and B (target)")
    ax.axvline(thr, color="#d62728", lw=2.0, label=f"gate threshold = {thr}")
    ax.set_yticks([])
    ax.set_xlabel("gate input  S = (a+b)/(2(1+cos))  ≈  number of features present")
    ax.set_title(f"Magnitude separation at cos(q_A,q_B) = {float(cos_choice):.1f}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_and · pass_6 — magnitude AND head\n"
        "A single hand-built attention head implements logical AND by "
        "**thresholding the total feature magnitude** rather than multiplying "
        "per-feature estimates, so the boundary survives full superposition "
        "(cos → 1) where a product gate collapses."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            run_dd = gr.Dropdown(
                choices=_run_choices(),
                value=(_run_choices()[0] if _run_choices() else None),
                label="result run", interactive=True,
            )
            summary = gr.Markdown(_summary(None))
            robust_fig = gr.Plot(label="AND sharpness vs superposition")
            cos_pick = gr.Radio(choices=["0.0", "1.0"], value="0.0",
                                label="cosine snapshot", interactive=True)
            sep_fig = gr.Plot(label="magnitude separation")

            def _refresh(name, cc):
                return robustness_plot(name), _summary(name), separation_plot(name, cc)

            run_dd.change(_refresh, inputs=[run_dd, cos_pick],
                          outputs=[robust_fig, summary, sep_fig])
            cos_pick.change(separation_plot, inputs=[run_dd, cos_pick], outputs=sep_fig)
            demo.load(_refresh, inputs=[run_dd, cos_pick],
                      outputs=[robust_fig, summary, sep_fig])

        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
