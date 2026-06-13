"""Gradio app for attention_matmul / pass_3.

Demo tab tells the mechanism-selection story for the chosen run/condition:

  1. THREE heatmaps — true attention vs the winning `softmax` attribution
     (visually identical, i.e. perfect recovery) vs the `no_softmax` linear
     strawman (visibly wrong). Eyeball check of the claim, cell by cell.
  2. ATTRIBUTION-KL bar chart — every candidate mechanism scored against the
     ground truth (lower = better). softmax wins; no_softmax / linear_taylor /
     wrong_temp / uniform all lose.
  3. OPERATING-RANGE line chart — fidelity vs input-magnitude multiplier across
     two orders of magnitude. softmax holds at ≈1; the cheap linear_taylor
     surrogate breaks as logits grow — a located breaking point.
  4. CAUSAL bars — necessity (ablate top-attributed key vs random) and
     sufficiency (top-k reconstruction MSE vs random-k).

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
KL_SERIES = ["softmax", "no_softmax", "linear_taylor", "wrong_temp", "uniform"]

# Plasma-ish colormap anchors for the heatmaps (no matplotlib dependency).
_ANCHORS = np.array(
    [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]],
    dtype=np.float64,
)


def _list_runs():
    if not os.path.isdir(RESULTS):
        return []
    runs = [
        d for d in os.listdir(RESULTS)
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


def _empty(cols):
    return pd.DataFrame({c: [] for c in cols})


def update(run, cond):
    empty_kl = _empty(["condition", "mechanism", "kl"])
    empty_op = _empty(["scale", "mechanism", "fidelity"])
    empty_ab = _empty(["condition", "series", "change"])
    empty_su = _empty(["k", "series", "mse"])
    if not run or not os.path.isfile(os.path.join(RESULTS, run, "summary.json")):
        return None, None, None, "No run found — execute `main.py` first.", \
            empty_kl, empty_op, empty_ab, empty_su

    s = _load_summary(run)
    if cond not in s["heatmaps"]:
        cond = s["conditions"][0]

    hm = s["heatmaps"][cond]
    true_img = _colorize(np.load(os.path.join(RESULTS, run, hm["true"])))
    method_img = _colorize(np.load(os.path.join(RESULTS, run, hm["method"])))
    straw_img = _colorize(np.load(os.path.join(RESULTS, run, hm["straw"])))

    pc = s["per_condition"][cond]
    kl = pc["hyp_kl"]
    md = (
        f"### `{cond}` — example head (b=0, h=0)\n"
        f"- **Attribution KL** (lower = better): "
        f"softmax **{kl['softmax']:.4f}** · no_softmax {kl['no_softmax']:.4f} · "
        f"linear_taylor {kl['linear_taylor']:.4f} · wrong_temp {kl['wrong_temp']:.4f} · "
        f"uniform {kl['uniform']:.4f}\n"
        f"- **Output reconstruction MSE** (mechanism @V vs true O): "
        f"softmax **{pc['hyp_out']['softmax']:.4f}** · "
        f"no_softmax {pc['hyp_out']['no_softmax']:.4f} · "
        f"uniform {pc['hyp_out']['uniform']:.4f}\n"
        f"- **Row-sum MAE** (convexity of softmax attribution): {pc['rowsum_mae']:.2e}\n"
        f"- **Necessity** (mean ‖ΔO‖ on key removal): top-attributed "
        f"**{pc['abl_top']:.3f}** vs random {pc['abl_random']:.3f}"
    )

    # KL bars across conditions × mechanisms.
    kl_rows = []
    for c in s["conditions"]:
        h = s["per_condition"][c]["hyp_kl"]
        for m in KL_SERIES:
            kl_rows.append({"condition": c, "mechanism": m, "kl": float(h[m])})

    # Operating-range lines.
    op = s["operating_range"]
    op_rows = []
    for name, fids in op["fidelity"].items():
        for sc, fd in zip(op["scales"], fids):
            op_rows.append({"scale": float(sc), "mechanism": name, "fidelity": float(fd)})

    # Causal: necessity bars + sufficiency lines.
    ab_rows, su_rows = [], []
    for c in s["conditions"]:
        p = s["per_condition"][c]
        ab_rows += [
            {"condition": c, "series": "top-key removed", "change": p["abl_top"]},
            {"condition": c, "series": "random-key removed", "change": p["abl_random"]},
        ]
    for k, mt, mr in zip(pc["suff_k"], pc["suff_top_mse"], pc["suff_rand_mse"]):
        su_rows += [
            {"k": int(k), "series": "top-k keys", "mse": float(mt)},
            {"k": int(k), "series": "random-k keys", "mse": float(mr)},
        ]

    return (
        true_img, method_img, straw_img, md,
        pd.DataFrame(kl_rows), pd.DataFrame(op_rows),
        pd.DataFrame(ab_rows), pd.DataFrame(su_rows),
    )


_runs = _list_runs()
_default_run = _runs[0] if _runs else None

with gr.Blocks(title="attention_matmul / pass_3") as demo:
    gr.Markdown(
        "# attention_matmul — mechanism selection\n"
        "The true attribution is `softmax(QK^T/√d)` **by construction**, so the "
        "interp question is not *what* the function is but *which mechanism* it is "
        "and *how we know*. We score several candidate mechanisms — `softmax` "
        "(the claim), `no_softmax`, `linear_taylor`, `wrong_temp`, `uniform` — by "
        "attribution KL, output reconstruction, and causal ablation, and map where "
        "the cheap linear surrogate breaks."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=_runs, value=_default_run, label="Run", interactive=True)
            cond_dd = gr.Dropdown(
                choices=CONDITIONS, value="cos_0p3",
                label="Condition (qk_alignment)", interactive=True,
            )
        with gr.Row():
            true_im = gr.Image(label="True attention softmax(QK^T/√d)", type="numpy")
            method_im = gr.Image(label="softmax attribution (the claim)", type="numpy")
            straw_im = gr.Image(label="no_softmax strawman (wrong)", type="numpy")
        metrics_md = gr.Markdown()

        kl_plot = gr.BarPlot(
            x="condition", y="kl", color="mechanism",
            title="Attribution KL vs ground truth (lower = better) — softmax wins",
            y_title="KL", x_title="qk_alignment",
        )
        op_plot = gr.LinePlot(
            x="scale", y="fidelity", color="mechanism",
            title="Operating range: fidelity vs input-magnitude multiplier (canonical cos_0p3)",
            x_title="logit-scale multiplier (×)", y_title="attribution fidelity",
        )
        with gr.Row():
            abl_plot = gr.BarPlot(
                x="condition", y="change", color="series",
                title="Necessity: mean ‖ΔO‖ when a key is removed",
                y_title="output change", x_title="qk_alignment",
            )
            suff_plot = gr.LinePlot(
                x="k", y="mse", color="series",
                title="Sufficiency: output MSE reconstructed from k keys (selected condition)",
                x_title="k keys kept", y_title="reconstruction MSE",
            )

        ev_inputs = [run_dd, cond_dd]
        ev_outputs = [true_im, method_im, straw_im, metrics_md,
                      kl_plot, op_plot, abl_plot, suff_plot]
        run_dd.change(update, ev_inputs, ev_outputs)
        cond_dd.change(update, ev_inputs, ev_outputs)
        demo.load(update, ev_inputs, ev_outputs)

    with gr.Tab("Benchmark"):
        gr.Markdown("## Cross-attempt leaderboard & metric history")
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
