"""Gradio app for attention_sat / pass_3.

Demo tab: shows (1) the detection contrast — softmax concentration rises and
separates the saturated regime while the no-exp ablation stays flat (fails);
(2) the measured gradient collapse that *defines* saturation; (3) the attention
heatmap at a chosen logit scale. Benchmark tab: shared leaderboard panel.
"""
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel

HERE = Path(__file__).parent
GOAL_DIR = HERE.parent
RESULTS = HERE / "results"


def _run_dirs():
    if not RESULTS.exists():
        return []
    return sorted((d for d in RESULTS.iterdir() if (d / "viz.json").exists()),
                  key=lambda d: d.name)


def _load_viz(run_name):
    if not run_name:
        runs = _run_dirs()
        if not runs:
            return None
        run = runs[-1]
    else:
        run = RESULTS / run_name
    f = run / "viz.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _fig_detection(viz):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if viz is None:
        ax.text(0.5, 0.5, "no run found", ha="center")
        return fig
    s = viz["scales"]
    r, st = viz["real"], viz["strawman"]
    ax.plot(s, r["concentration"], "o-", color="#1f77b4",
            label=f"softmax  (AUROC={r['auroc']:.2f})")
    ax.plot(s, st["concentration"], "s--", color="#d62728",
            label=f"no-exp ablation (AUROC={st['auroc']:.2f})")
    ax.axvline(viz["threshold"], color="gray", ls=":", label="saturated ⇔ scale≥10")
    ax.set_xscale("log")
    ax.set_xlabel("logit scale (inverse temperature)")
    ax.set_ylabel("attention concentration  Σp²  (= saturation score)")
    ax.set_title("Saturation detection: exp is the mechanism")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_gradient(viz):
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    if viz is None:
        ax.text(0.5, 0.5, "no run found", ha="center")
        return fig
    s = viz["scales"]
    g = viz["real"]["grad_norm"]
    ax.plot(s, g, "o-", color="#2ca02c")
    ax.axvline(viz["threshold"], color="gray", ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("logit scale")
    ax.set_ylabel("mean |∂‖attn·v‖² / ∂logits|  (autograd)")
    ax.set_title("Gradient collapse = mechanistic saturation\n(softmax Jacobian → 0)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    return fig


def _fig_heatmap(viz, idx):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if viz is None:
        ax.text(0.5, 0.5, "no run found", ha="center")
        return fig
    idx = int(max(0, min(idx, len(viz["scales"]) - 1)))
    mat = np.array(viz["mean_attn"][idx])
    im = ax.imshow(mat, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_title(f"softmax weights (scale={viz['scales'][idx]:g})")
    ax.set_xlabel("key position")
    ax.set_ylabel("query position")
    fig.colorbar(im, ax=ax, label="weight")
    fig.tight_layout()
    return fig


def _summary(viz):
    if viz is None:
        return {"status": "no run found — run main.py first"}
    r, st = viz["real"], viz["strawman"]
    return {
        "softmax_auroc": round(r["auroc"], 4),
        "no_exp_ablation_auroc": round(st["auroc"], 4),
        "ablation_drop": round(r["auroc"] - st["auroc"], 4),
        "concentration_linear_regime(scale=0.1)": round(r["concentration"][0], 4),
        "concentration_saturated(scale=100)": round(r["concentration"][-1], 4),
        "grad_norm_linear(scale=0.1)": float(f"{r['grad_norm'][0]:.3e}"),
        "grad_norm_saturated(scale=100)": float(f"{r['grad_norm'][-1]:.3e}"),
        "claim": "exp drives saturation; ablating it (no-exp head) is scale-invariant and fails to detect.",
    }


def _refresh(run_name, idx):
    viz = _load_viz(run_name)
    return (_fig_detection(viz), _fig_gradient(viz),
            _fig_heatmap(viz, idx), _summary(viz))


def _update_heatmap(run_name, idx):
    return _fig_heatmap(_load_viz(run_name), idx)


_runs = [d.name for d in _run_dirs()]
_default = _runs[-1] if _runs else None

with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_sat · pass_3 — saturation = softmax-Jacobian collapse\n"
        "`saturation_score = Σp²` (one minus the softmax Jacobian trace = the "
        "vanishing-gradient quantity). The **no-exp ablation** is scale-invariant "
        "and cannot saturate — it is the failing strawman that isolates `exp`."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=_runs, value=_default, label="run")
            scale_idx = gr.Slider(0, 6, value=4, step=1,
                                  label="scale index (0=0.1 … 4=10 … 6=100)")
            reload_btn = gr.Button("reload", variant="primary")
        with gr.Row():
            det_plot = gr.Plot(label="detection contrast")
            grad_plot = gr.Plot(label="gradient collapse")
        with gr.Row():
            heat_plot = gr.Plot(label="attention heatmap")
            summary = gr.JSON(label="summary")

        reload_btn.click(_refresh, [run_dd, scale_idx],
                         [det_plot, grad_plot, heat_plot, summary])
        run_dd.change(_refresh, [run_dd, scale_idx],
                      [det_plot, grad_plot, heat_plot, summary])
        scale_idx.change(_update_heatmap, [run_dd, scale_idx], [heat_plot])
        demo.load(_refresh, [run_dd, scale_idx],
                  [det_plot, grad_plot, heat_plot, summary])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
