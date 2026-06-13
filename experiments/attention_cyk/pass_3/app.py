import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)
RESULTS = os.path.join(ATTEMPT_DIR, "results")


def list_runs():
    if not os.path.isdir(RESULTS):
        return []
    runs = [
        d
        for d in glob.glob(os.path.join(RESULTS, "*"))
        if os.path.exists(os.path.join(d, "demo.json"))
    ]
    runs.sort(reverse=True)
    return runs


def run_choices():
    return [(os.path.basename(r), r) for r in list_runs()]


def load_demo(run_dir):
    if not run_dir:
        return None
    try:
        with open(os.path.join(run_dir, "demo.json")) as f:
            return json.load(f)
    except Exception:
        return None


def example_labels(data):
    if not data:
        return []
    return [f"{k}: {ex['label']}" for k, ex in enumerate(data["examples"])]


def _blank(msg):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


def detail_fig(run_dir, label):
    data = load_demo(run_dir)
    if not data or not data["examples"]:
        return _blank("no run data yet — run main.py")
    idx = 0
    if label:
        try:
            idx = int(str(label).split(":")[0])
        except Exception:
            idx = 0
    idx = max(0, min(idx, len(data["examples"]) - 1))
    ex = data["examples"][idx]
    depths = ex["depths"]
    n = len(depths) - 1
    i, j = ex["i"], ex["j"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
    xs = list(range(n + 1))
    ax1.plot(xs, depths, marker="o", color="#333333")
    ax1.axhline(depths[i], ls="--", color="#1f77b4", label=f"D(i) = {depths[i]}")
    ax1.axvspan(i, j, color="#cfe8ff", alpha=0.4, label=f"cell ({i},{j})")
    ax1.set_title(f"bracket-depth profile   {ex['seq_str']}   [{ex['cell_type']}]")
    ax1.set_xlabel("position")
    ax1.set_ylabel("depth D(p)")
    ax1.set_xticks(xs)
    ax1.legend(fontsize=8, loc="upper right")

    cands = ex["candidates"]
    scores = ex["scores"]
    correct = set(ex["correct"])
    colors = ["#2ca02c" if k in correct else "#cccccc" for k in cands]
    ax2.bar([str(k) for k in cands], scores, color=colors)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("single-head attention over split points (green = CYK-correct)")
    ax2.set_xlabel("split position k")
    ax2.set_ylabel("attention prob")
    fig.tight_layout()
    return fig


def summary_fig(run_dir):
    data = load_demo(run_dir)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if not data:
        return _blank("no run data yet")
    s = data["summary"]
    names = ["full\n(1 head)", "no\nposition", "depth\nablated", "uniform"]
    keys = ["full", "no_position", "depth_ablated", "uniform"]
    vals = [s[k] for k in keys]
    bars = ax.bar(names, vals, color=["#2ca02c", "#ff7f0e", "#d62728", "#888888"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("split accuracy (canonical batch)")
    ax.set_title("headline + two ablations")
    fig.tight_layout()
    return fig


def span_fig(run_dir):
    data = load_demo(run_dir)
    fig, ax = plt.subplots(figsize=(6, 4))
    if not data or not data.get("per_span"):
        return _blank("no run data yet")
    ps = data["per_span"]
    ax.plot(ps["span"], ps["full"], marker="o", color="#2ca02c", label="full circuit")
    ax.plot(ps["span"], ps["uniform"], marker="s", color="#888888", label="uniform")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("span length j - i")
    ax.set_ylabel("split accuracy")
    ax.set_title("operating range across span length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def on_run_change(run_dir):
    data = load_demo(run_dir)
    labels = example_labels(data)
    first = labels[0] if labels else None
    return (
        gr.update(choices=labels, value=first),
        detail_fig(run_dir, first),
        summary_fig(run_dir),
        span_fig(run_dir),
    )


def init():
    runs = run_choices()
    default = runs[0][1] if runs else None
    dd, detail, summ, span = on_run_change(default)
    return (
        gr.update(choices=runs, value=default),
        dd,
        detail,
        summ,
        span,
    )


with gr.Blocks(title="attention_cyk · pass_3") as demo:
    gr.Markdown(
        "# attention_cyk — one pure-attention head for the CYK split\n"
        "A **causal counting head** computes bracket depth `D(p)`; a **single "
        "split head** scores split `k` by "
        "`-T·(D(k)-D(i))² + β·(D(i)+0.5-D(k))·k` — a genuine linear QK score. "
        "The quadratic snaps onto the depth level `D(i)`; the position term "
        "picks the *latest* such point for `S→S S` / `X→S R` cells and the "
        "*earliest* min-level point (`k=i+1`) for wrapped `S→L X` cells — **no "
        "Python routing, one head**."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(label="run", choices=run_choices(), interactive=True)
            ex_dd = gr.Dropdown(label="example cell", choices=[], interactive=True)
        with gr.Row():
            detail_plot = gr.Plot(label="depth profile → attention over splits")
        with gr.Row():
            summary_plot = gr.Plot(label="headline + ablations")
            span_plot = gr.Plot(label="operating range")
        gr.Markdown(
            "**How to read it.** Top: the string's bracket-depth profile; the "
            "dashed line is the cell's start depth `D(i)`. Bottom: the head's "
            "attention over candidate splits — green = CYK-correct. Cycle the "
            "example dropdown through the three cell types: for `S→S S` and "
            "`X→S R` cells the head lands on the latest depth-`D(i)` crossing; "
            "for wrapped `S→L X` cells (no crossing) it lands on `k=i+1`. The "
            "bars compare the full single head against **no-position** (β=0, "
            "pure depth-matching) and **depth-ablated** (`D:=0`) controls."
        )

        run_dd.change(
            on_run_change,
            inputs=run_dd,
            outputs=[ex_dd, detail_plot, summary_plot, span_plot],
        )
        ex_dd.change(detail_fig, inputs=[run_dd, ex_dd], outputs=detail_plot)
        demo.load(
            init,
            inputs=None,
            outputs=[run_dd, ex_dd, detail_plot, summary_plot, span_plot],
        )

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
