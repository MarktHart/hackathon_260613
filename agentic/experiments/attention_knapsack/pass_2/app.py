"""Gradio app for attention_knapsack / pass_2.

Demo tab: shows the central claim -- our attention-guided selection circuit
closes the optimality gap that the greedy ratio heuristic leaves open, across
the whole capacity sweep, while staying 100% feasible. Plus a single-instance
illustration of an improving 1-exchange the greedy heuristic cannot find.

Benchmark tab: cross-attempt leaderboard / history via benchmark_panel.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"


def _opt(gap):
    return max(0.0, min(1.0, 1.0 - float(gap)))


def list_runs():
    if not RESULTS.is_dir():
        return []
    runs = [p.name for p in RESULTS.iterdir()
            if (p / "benchmark.json").is_file()]
    return sorted(runs, reverse=True)


def _load(run_id):
    bench = json.loads((RESULTS / run_id / "benchmark.json").read_text())
    ex_path = RESULTS / run_id / "example.json"
    example = json.loads(ex_path.read_text()) if ex_path.is_file() else None
    return bench, example


def sweep_plot(run_id):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not run_id:
        ax.text(0.5, 0.5, "no run selected", ha="center", va="center")
        ax.set_axis_off()
        return fig
    bench, _ = _load(run_id)
    payload = bench["payload"]
    fracs = [r["capacity_frac"] for r in payload["sweep"]]
    model = [_opt(r["optimality_gap"]) for r in payload["sweep"]]
    base = [_opt(r["optimality_gap"]) for r in payload["baseline_sweep"]]
    ax.plot(fracs, model, marker="o", lw=2, color="#1f77b4",
            label="Attention circuit (ours)")
    ax.plot(fracs, base, marker="s", lw=2, ls="--", color="#d62728",
            label="Greedy ratio baseline")
    ax.axhline(1.0, color="grey", lw=1, alpha=0.5, label="exact optimum")
    ax.fill_between(fracs, base, model, color="#1f77b4", alpha=0.15)
    ax.set_xlabel("capacity fraction")
    ax.set_ylabel("optimality  (1 − gap),  higher = better")
    ax.set_title("Fraction of optimal knapsack value attained vs. greedy")
    lo = min(base) - 0.004
    ax.set_ylim(lo, 1.001)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def example_plot(run_id):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not run_id:
        ax.text(0.5, 0.5, "no run selected", ha="center", va="center")
        ax.set_axis_off()
        return fig
    _, ex = _load(run_id)
    if ex is None:
        ax.text(0.5, 0.5, "no example saved for this run",
                ha="center", va="center")
        ax.set_axis_off()
        return fig
    labels = ["Greedy", "Ours\n(attention)", "Optimal"]
    vals = [ex["greedy_value"], ex["refined_value"], ex["optimal_value"]]
    colors = ["#d62728", "#1f77b4", "#2ca02c"]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("total selected value")
    ax.set_title(f"One canonical instance (capacity = {ex['capacity']:.0f}): "
                 "our 1-exchange beats greedy, reaches optimum")
    ax.set_ylim(0, max(vals) * 1.15)
    fig.tight_layout()
    return fig


def summary_md(run_id):
    if not run_id:
        return "No runs yet. Run `main.py` to populate `results/`."
    bench, ex = _load(run_id)
    m = bench["metrics"]
    head = m.get("knapsack_optimality_robustness", float("nan"))
    can = m.get("knapsack_optimality_canonical", float("nan"))
    feas = m.get("knapsack_feasible_canonical", float("nan"))
    base = m.get("linear_baseline_optimality_canonical", float("nan"))
    lift = m.get("lift_over_linear_baseline", float("nan"))
    lines = [
        f"### Run `{run_id}`",
        "",
        f"- **Headline — sweep optimality robustness:** `{head:.4f}`",
        f"- Canonical optimality (cap 0.5): `{can:.4f}`  "
        f"vs greedy `{base:.4f}`  →  **lift `{lift:+.4f}`**",
        f"- Canonical feasible rate: `{feas:.3f}`  (selections respect capacity)",
    ]
    if ex is not None:
        def fmt(sel):
            return "".join("█" if s else "·" for s in sel)
        lines += [
            "",
            "**Item-level view of the example instance** "
            "(`█` = item selected):",
            "",
            "| solution | selection (16 items) | value |",
            "|---|---|---|",
            f"| greedy | `{fmt(ex['greedy_sel'])}` | {ex['greedy_value']:.0f} |",
            f"| ours | `{fmt(ex['refined_sel'])}` | {ex['refined_value']:.0f} |",
            f"| optimal | `{fmt(ex['optimal_sel'])}` | {ex['optimal_value']:.0f} |",
        ]
    return "\n".join(lines)


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_knapsack — pass_2\n"
        "Hand-built **attention-guided selection circuit**: greedy-ratio init + "
        "hard-attention 1-exchange local search. Every applied move is a "
        "value-improving, capacity-feasible argmax, so the result is feasible "
        "by construction and dominates the greedy baseline."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(choices=list_runs(),
                                     value=(list_runs()[0] if list_runs() else None),
                                     label="results run (latest first)")
                refresh = gr.Button("Refresh runs", size="sm")
            summary = gr.Markdown()
            with gr.Row():
                sweep_fig = gr.Plot(label="Optimality across capacity sweep")
                ex_fig = gr.Plot(label="Single-instance comparison")

            def _update(run_id):
                return summary_md(run_id), sweep_plot(run_id), example_plot(run_id)

            def _refresh():
                runs = list_runs()
                sel = runs[0] if runs else None
                return gr.update(choices=runs, value=sel)

            run_dd.change(_update, inputs=run_dd,
                          outputs=[summary, sweep_fig, ex_fig])
            refresh.click(_refresh, outputs=run_dd).then(
                _update, inputs=run_dd, outputs=[summary, sweep_fig, ex_fig])
            demo.load(_update, inputs=run_dd,
                      outputs=[summary, sweep_fig, ex_fig])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
