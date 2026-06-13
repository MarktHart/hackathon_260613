"""Gradio app for attention_span / pass_5.

Demo tab: the attention-on-target vs log2(distance) curves for the hand-built
content head, the positional-only strawman, and the ablated head, plus an
interactive ALiBi-slope slider that shows how a distance penalty shrinks the
effective span. Benchmark tab: cross-attempt leaderboard + history.
"""

import json
import pathlib

import numpy as np
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = pathlib.Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS_DIR = ATTEMPT_DIR / "results"
SEQ_LEN = 512
UNIFORM = 1.0 / SEQ_LEN
DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def list_runs():
    if not RESULTS_DIR.is_dir():
        return []
    runs = [p.name for p in RESULTS_DIR.iterdir() if (p / "curves.json").exists()]
    return sorted(runs, reverse=True)


def load_curves(run_id):
    """Load saved contrast curves for a run; fall back to a synthetic shape."""
    runs = list_runs()
    if run_id in runs:
        data = json.loads((RESULTS_DIR / run_id / "curves.json").read_text())
        return data
    if runs:
        data = json.loads((RESULTS_DIR / runs[0] / "curves.json").read_text())
        return data
    # Fallback (no run yet): illustrative numbers only.
    return {
        "distances": DISTANCES,
        "content_means": [0.99] * len(DISTANCES),
        "positional_means": [float(np.exp(-0.05 * d)) for d in DISTANCES],
        "ablation_means": [UNIFORM] * len(DISTANCES),
        "content_auc": 0.99,
        "positional_auc": 0.02,
        "ablation_auc": UNIFORM,
        "uniform_baseline": UNIFORM,
    }


def positional_curve(slope):
    """Analytic ALiBi positional-head attention on the needle at each distance.

    attn(d) = exp(-slope*d) / sum_{j=0..L-1} exp(-slope*j).
    """
    j = np.arange(SEQ_LEN)
    denom = np.exp(-slope * j).sum()
    return [float(np.exp(-slope * d) / denom) for d in DISTANCES]


def main_plot(run_id):
    data = load_curves(run_id)
    d = np.array(data["distances"], dtype=float)
    x = np.log2(d)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, data["content_means"], "o-", lw=2.5, color="#1b9e77",
            label=f"content head (AUC={data['content_auc']:.3f})")
    ax.plot(x, data["positional_means"], "s--", lw=2, color="#d95f02",
            label=f"positional strawman (AUC={data['positional_auc']:.3f})")
    ax.plot(x, data["ablation_means"], "^:", lw=2, color="#7570b3",
            label=f"ablated content head (AUC={data['ablation_auc']:.3f})")
    ax.axhline(data["uniform_baseline"], color="gray", lw=1,
               label=f"uniform baseline (1/512={UNIFORM:.4f})")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in d])
    ax.set_xlabel("distance from query to target (log2 scale)")
    ax.set_ylabel("mean query→target attention")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Effective attention span: content addressing is distance-invariant")
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def slope_plot(slope):
    pos = positional_curve(slope)
    d = np.array(DISTANCES, dtype=float)
    x = np.log2(d)
    # Effective span = largest distance whose attention still beats uniform.
    span = 0
    for dist, a in zip(DISTANCES, pos):
        if a >= UNIFORM:
            span = dist
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, pos, "s--", lw=2, color="#d95f02", label=f"positional head (slope={slope:.3f})")
    ax.axhline(UNIFORM, color="gray", lw=1, label="uniform baseline (1/512)")
    ax.fill_between(x, UNIFORM, pos, where=np.array(pos) >= UNIFORM,
                    color="#d95f02", alpha=0.15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in d])
    ax.set_xlabel("distance from query to target (log2 scale)")
    ax.set_ylabel("mean query→target attention")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title(f"Positional-only span shrinks with the ALiBi slope — effective span ≈ {span}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# attention_span — content-addressed retrieval head (pass_5)")
    with gr.Tab("Demo"):
        gr.Markdown(
            "**Claim:** a hand-built single attention head whose Q/K projections align "
            "the query token (8888) with the needle token (9999) retrieves the needle at "
            "**any** distance — its span is limited by *content*, not *distance*. The curve "
            "is flat at ~0.997 across 1→256 (>2 orders of magnitude), so the robustness "
            "ratio (long/short) ≈ 1.0.\n\n"
            "The **positional strawman** (ALiBi distance penalty, no content) and the "
            "**ablated** head (needle key direction zeroed) both collapse — the ablation "
            "is the causal check that the span comes from the Q/K alignment."
        )
        run_dd = gr.Dropdown(choices=list_runs(), value=(list_runs() or [None])[0],
                             label="results run")
        main_fig = gr.Plot(label="attention-on-target sweep")
        run_dd.change(main_plot, inputs=run_dd, outputs=main_fig)

        gr.Markdown(
            "### Interactive: how a distance penalty bounds the span\n"
            "A content head ignores this knob. A **positional-only** head's reach is set by "
            "the ALiBi slope: slide it up and the effective span (last distance beating the "
            "uniform baseline) collapses toward the query."
        )
        slope_sl = gr.Slider(0.001, 0.2, value=0.05, step=0.001, label="ALiBi slope m")
        slope_fig = gr.Plot(label="positional-head decay")
        slope_sl.change(slope_plot, inputs=slope_sl, outputs=slope_fig)

        demo.load(main_plot, inputs=run_dd, outputs=main_fig)
        demo.load(slope_plot, inputs=slope_sl, outputs=slope_fig)

    with gr.Tab("Benchmark"):
        gr.Markdown("## Cross-attempt benchmark history")
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
