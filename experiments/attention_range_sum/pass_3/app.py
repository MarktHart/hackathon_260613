"""Gradio app for attention_range_sum / pass_3.

Demo tab:
  * MSE vs window size k for the full head vs two ablations vs the constant
    baseline — the headline faithfulness chart.
  * An interactive single-query inspector: pick k and the window start, see the
    softmax attention weights over all 64 positions (window highlighted) and the
    predicted vs true sum.
Benchmark tab:
  * the shared agentic leaderboard across every attempt at this goal.
"""

import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import load_task, benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

task = load_task(__file__)
batch = task.generate(42)
INPUT_IDS = np.asarray(batch.input_ids)
L = int(INPUT_IDS.shape[0])
RANGE_LENS = [2, 4, 8, 16, 32]
GAIN = 30.0


def list_runs():
    if not RESULTS.exists():
        return []
    return sorted([p.name for p in RESULTS.iterdir() if p.is_dir()], reverse=True)


def attn_weights(start, end, select=True, gain=GAIN):
    pos = np.arange(L)
    if select:
        scores = gain * ((pos >= start) & (pos < end)).astype(float)
    else:
        scores = np.zeros(L)
    s = scores - scores.max()
    e = np.exp(s)
    return e / e.sum()


def mse_chart(run_name):
    if not run_name:
        return None
    ab_path = RESULTS / run_name / "ablation.json"
    if not ab_path.exists():
        return None
    ab = json.loads(ab_path.read_text())
    ks = ab["range_lens"]
    floor = 1e-8

    def series(key):
        return [max(float(ab[key][str(k)]), floor) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.plot(ks, series("full"), "o-", lw=2.5, label="full head (select + scale)")
    ax.plot(ks, series("no_selection"), "s--", lw=2,
            label="ablate window selection")
    ax.plot(ks, series("no_scaling"), "^--", lw=2, label="ablate length scaling")
    ax.plot(ks, series("baseline"), "x:", lw=2, color="black",
            label="constant baseline (variance)")
    ax.set_yscale("log")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("window size k")
    ax.set_ylabel("MSE  (log, floored at 1e-8)")
    ax.set_title("Range-sum MSE vs window size — mechanism vs ablations")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = ATTEMPT_DIR / "_mse_chart.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def demo_attention(k, start):
    k = int(k)
    start = int(max(0, min(int(start), L - k)))
    end = start + k
    w = attn_weights(start, end)
    pred = float((w * INPUT_IDS).sum() * k)
    true = float(INPUT_IDS[start:end].sum())

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    colors = ["#d62728" if (start <= p < end) else "#cfcfcf" for p in range(L)]

    ax0.bar(range(L), w, color=colors)
    ax0.set_ylabel("attention weight")
    ax0.set_title(
        f"Softmax window selection — window [{start}, {end})  (k={k})"
    )
    ax1.bar(range(L), INPUT_IDS, color=colors)
    ax1.set_ylabel("token value")
    ax1.set_xlabel("position")
    ax1.set_title("Token values (red = inside window, summed by the head)")
    fig.tight_layout()
    path = ATTEMPT_DIR / "_demo_attn.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)

    txt = (
        f"**Predicted sum:** {pred:.4f}  |  **True sum:** {true:.0f}  |  "
        f"**abs error:** {abs(pred - true):.4f}"
    )
    return str(path), txt


_runs = list_runs()
_default_run = _runs[0] if _runs else None

with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_range_sum — pass_3 (hand-built head)\n"
        "A single attention head selects the window `[start, end)` with a "
        "softmax over one-hot positional keys, averages the token values, and a "
        "length-scaled readout (`× k`) recovers the **sum**. No MLP, no training."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=_runs, value=_default_run, label="Run (latest first)"
            )
        mse_img = gr.Image(label="MSE vs window size — mechanism vs ablations")
        gr.Markdown(
            "### Inspect the head on a single query\n"
            "The window selection is a genuine Q·K softmax: ablating it "
            "collapses the head onto the constant baseline (see chart above), "
            "and ablating the `× k` readout makes it predict the mean, not the "
            "sum."
        )
        with gr.Row():
            k_dd = gr.Dropdown(choices=RANGE_LENS, value=8, label="window size k")
            start_sl = gr.Slider(
                0, L - 2, value=10, step=1, label="window start"
            )
        attn_img = gr.Image(label="Attention weights & token values")
        pred_md = gr.Markdown()

        run_dd.change(mse_chart, inputs=run_dd, outputs=mse_img)
        k_dd.change(demo_attention, inputs=[k_dd, start_sl],
                    outputs=[attn_img, pred_md])
        start_sl.change(demo_attention, inputs=[k_dd, start_sl],
                        outputs=[attn_img, pred_md])

        demo.load(mse_chart, inputs=run_dd, outputs=mse_img)
        demo.load(demo_attention, inputs=[k_dd, start_sl],
                  outputs=[attn_img, pred_md])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
