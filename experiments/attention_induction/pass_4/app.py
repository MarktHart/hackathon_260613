"""Gradio app for pass_4 — hand-built induction circuit.

Demo tab: per-distance accuracy bar chart comparing
  (1) the full induction circuit,
  (2) the same circuit with the previous-token head ablated, and
  (3) the uniform baseline.
The collapse from (1) to (2) is the faithfulness story: induction is *caused*
by the layer-0 previous-token head feeding the layer-1 match.

Benchmark tab: the shared cross-attempt leaderboard.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


def _list_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = [p.name for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "demo.json").exists()]
    return sorted(runs, reverse=True)


def _load_demo(run_id):
    if not run_id:
        return None
    path = RESULTS_DIR / run_id / "demo.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _get(by_dist, d):
    if str(d) in by_dist:
        return by_dist[str(d)]
    return by_dist.get(d, 0.0)


def _plot(run_id):
    data = _load_demo(run_id)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if data is None:
        ax.text(0.5, 0.5, "No run found.\nRun main.py first.",
                ha="center", va="center")
        ax.axis("off")
        return fig

    dists = sorted(int(d) for d in data["full"]["by_distance"].keys())
    full = [_get(data["full"]["by_distance"], d) for d in dists]
    abl = [_get(data["ablated_prev_head"]["by_distance"], d) for d in dists]
    uniform = data["uniform_baseline_accuracy"]

    x = range(len(dists))
    w = 0.38
    ax.bar([i - w / 2 for i in x], full, width=w,
           label="Full induction circuit", color="#2b8cbe")
    ax.bar([i + w / 2 for i in x], abl, width=w,
           label="Prev-token head ablated", color="#d6604d")
    ax.axhline(uniform, ls="--", color="gray", lw=1.2,
               label=f"Uniform baseline ({uniform:.4f})")

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"P={d}" for d in dists])
    ax.set_xlabel("Occurrence distance P")
    ax.set_ylabel("Induction accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Induction accuracy vs copy distance\n(ablating the prev-token head collapses it)")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    return fig


def _summary(run_id):
    data = _load_demo(run_id)
    if data is None:
        return "No run found. Run `main.py` to generate results."
    f = data["full"]
    a = data["ablated_prev_head"]
    return (
        f"**Full circuit** — accuracy `{f['aggregate_accuracy']:.4f}`, "
        f"CE `{f['aggregate_ce_loss']:.4f}` nats\n\n"
        f"**Prev-token head ablated** — accuracy `{a['aggregate_accuracy']:.4f}`, "
        f"CE `{a['aggregate_ce_loss']:.4f}` nats\n\n"
        f"**Uniform baseline** — accuracy `{data['uniform_baseline_accuracy']:.4f}`\n\n"
        f"Knocking out the layer-0 previous-token head drops accuracy from "
        f"`{f['aggregate_accuracy']:.3f}` to `{a['aggregate_accuracy']:.3f}` — "
        f"the induction behaviour is *caused* by the circuit."
    )


_runs = _list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks() as demo:
    gr.Markdown("# Attention Induction — pass_4: hand-built 2-layer induction circuit")
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown(
                "A tiny **attention-only** transformer with hand-set weights: a "
                "layer-0 *previous-token head* feeds a layer-1 *induction head*. "
                "Logits flow through the real attention + unembedding — nothing is "
                "bypassed. The chart shows the circuit working across copy "
                "distances, and the **ablation** (removing the prev-token head) "
                "collapsing it to the uniform baseline."
            )
            run_dd = gr.Dropdown(choices=_runs, value=_default, label="Run")
            plot = gr.Plot(label="Per-distance accuracy")
            summ = gr.Markdown()

            run_dd.change(_plot, inputs=run_dd, outputs=plot)
            run_dd.change(_summary, inputs=run_dd, outputs=summ)
            demo.load(_plot, inputs=run_dd, outputs=plot)
            demo.load(_summary, inputs=run_dd, outputs=summ)

        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
