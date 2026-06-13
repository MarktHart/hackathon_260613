"""Gradio app for attention_matmul / pass_2.

Demo tab — for the selected run/condition: true attention vs the gradient-Jacobian
attribution as heatmaps, a KL-vs-baselines bar chart (method beats both the uniform
and the no-softmax strawman), and the causal ablation bar chart (removing the
top-attributed key collapses the output; removing a random key barely moves it).

Benchmark tab — the shared cross-attempt dashboard.
"""

import os
import json
import numpy as np
import pandas as pd
import gradio as gr

from agentic.experiments import benchmark_panel

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
GOAL_DIR = os.path.dirname(HERE)  # experiments/attention_matmul

CONDITIONS = ["orthogonal", "cos_0p3", "cos_0p7", "uniform"]

# Plasma-ish colormap anchors for the heatmaps (avoids a matplotlib dependency).
_ANCHORS = np.array(
    [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]],
    dtype=np.float64,
)


def _list_runs():
    if not os.path.isdir(RESULTS):
        return []
    runs = [
        d
        for d in os.listdir(RESULTS)
        if os.path.isfile(os.path.join(RESULTS, d, "summary.json"))
    ]
    return sorted(runs, reverse=True)


def _load_summary(run):
    with open(os.path.join(RESULTS, run, "summary.json")) as f:
        return json.load(f)


def _colorize(mat, upscale=10):
    m = np.asarray(mat, dtype=np.float64)
    mx = m.max() if m.max() > 0 else 1.0
    v = np.clip(m / mx, 0.0, 1.0)
    idx = v * (len(_ANCHORS) - 1)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(_ANCHORS) - 1)
    frac = (idx - lo)[..., None]
    col = _ANCHORS[lo] * (1 - frac) + _ANCHORS[hi] * frac
    img = col.astype(np.uint8)
    img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    return img


def _empty_kl():
    return pd.DataFrame({"condition": [], "series": [], "kl": []})


def _empty_abl():
    return pd.DataFrame({"condition": [], "series": [], "change": []})


def update(run, cond):
    if not run or not os.path.isfile(os.path.join(RESULTS, run, "summary.json")):
        return None, None, "No run found — execute `main.py` first.", _empty_kl(), _empty_abl()

    s = _load_summary(run)
    if cond not in s["heatmaps"]:
        cond = s["conditions"][0]

    true = np.load(os.path.join(RESULTS, run, s["heatmaps"][cond]["true"]))
    pred = np.load(os.path.join(RESULTS, run, s["heatmaps"][cond]["pred"]))
    true_img = _colorize(true)
    pred_img = _colorize(pred)

    pc = s["per_condition"][cond]
    md = (
        f"### `{cond}` — example head (b=0, h=0)\n"
        f"- **Attention KL** (lower = better): method **{pc['method_kl']:.4f}** · "
        f"uniform {pc['uniform_kl']:.4f} · no-softmax {pc['no_softmax_kl']:.4f}\n"
        f"- **Output MSE**: method **{pc['output_mse_method']:.4f}** · "
        f"uniform {pc['output_mse_uniform']:.4f}\n"
        f"- **Row-sum MAE** (convexity): {pc['rowsum_mae']:.2e}\n"
        f"- **Causal ablation** (mean ‖ΔO‖): remove top-attributed key "
        f"**{pc['abl_top']:.3f}** vs remove random key {pc['abl_random']:.3f}"
    )

    kl_rows, abl_rows = [], []
    for c in s["conditions"]:
        p = s["per_condition"][c]
        kl_rows += [
            {"condition": c, "series": "method", "kl": p["method_kl"]},
            {"condition": c, "series": "uniform", "kl": p["uniform_kl"]},
            {"condition": c, "series": "no_softmax", "kl": p["no_softmax_kl"]},
        ]
        abl_rows += [
            {"condition": c, "series": "top-key removed", "change": p["abl_top"]},
            {"condition": c, "series": "random-key removed", "change": p["abl_random"]},
        ]
    return true_img, pred_img, md, pd.DataFrame(kl_rows), pd.DataFrame(abl_rows)


_runs = _list_runs()
_default_run = _runs[0] if _runs else None

with gr.Blocks(title="attention_matmul / pass_2") as demo:
    gr.Markdown(
        "# attention_matmul — gradient-Jacobian attribution\n"
        "Attribution[i,j] is computed as the **causal sensitivity** d O_i / d V_j "
        "via GPU autograd — not by copying the generator's softmax. The charts show "
        "it recovers the true query-key pathway and that the keys it flags are the "
        "ones that causally drive the output."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=_runs, value=_default_run, label="Run", interactive=True
            )
            cond_dd = gr.Dropdown(
                choices=CONDITIONS, value="cos_0p3", label="Condition (qk_alignment)",
                interactive=True,
            )
        with gr.Row():
            true_im = gr.Image(label="True attention softmax(QK^T/√d)", type="numpy")
            pred_im = gr.Image(label="Predicted attribution (∂O/∂V Jacobian)", type="numpy")
        metrics_md = gr.Markdown()
        kl_plot = gr.BarPlot(
            x="condition", y="kl", color="series",
            title="Attention KL vs ground truth (lower = better) — method vs baselines",
            y_title="KL", x_title="qk_alignment",
        )
        abl_plot = gr.BarPlot(
            x="condition", y="change", color="series",
            title="Causal ablation: mean ‖ΔO‖ when a key is removed",
            y_title="output change", x_title="qk_alignment",
        )

        ev_inputs = [run_dd, cond_dd]
        ev_outputs = [true_im, pred_im, metrics_md, kl_plot, abl_plot]
        run_dd.change(update, ev_inputs, ev_outputs)
        cond_dd.change(update, ev_inputs, ev_outputs)
        demo.load(update, ev_inputs, ev_outputs)

    with gr.Tab("Benchmark"):
        gr.Markdown("## Cross-attempt leaderboard & metric history")
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
