"""Gradio app for the hand-built k-th position selection head.

Demo tab: for a chosen k, plot the mean attention distribution over positions
for the positional head vs the content-matching strawman vs uniform, with a
marker line at the true position k. The story: the positional head spikes at k;
content matching leaks onto spurious markers; uniform is flat.

Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

METHOD_STYLE = {
    "positional": ("#1f77b4", "hand-built positional head"),
    "content": ("#d62728", "content-matching strawman"),
    "uniform": ("#7f7f7f", "uniform baseline"),
}


def _list_runs() -> list[str]:
    if not RESULTS_DIR.exists():
        return []
    runs = [p.name for p in RESULTS_DIR.iterdir()
            if p.is_dir() and (p / "comparison.json").exists()]
    return sorted(runs, reverse=True)


def _load_comparison(run: str) -> dict | None:
    if not run:
        return None
    path = RESULTS_DIR / run / "comparison.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _plot(run: str, k: int):
    comp = _load_comparison(run)
    if comp is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no run found — execute main.py first",
                ha="center", va="center")
        ax.axis("off")
        return fig, "No data."

    L = comp.get("L", 32)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    summary_lines = [f"Target position k = {k}", ""]
    for name, (color, label) in METHOD_STYLE.items():
        recs = comp["methods"].get(name, [])
        rec = next((r for r in recs if r["k"] == k), None)
        if rec is None:
            continue
        ax.plot(range(L), rec["mean_attn"], color=color, label=label, marker=".")
        summary_lines.append(
            f"{label}: attn@k={rec['attn_at_k']:.3f}  "
            f"mean argmax={rec['attn_max_pos']:.2f}"
        )

    ax.axvline(k, color="black", linestyle="--", linewidth=1, label=f"true k={k}")
    ax.set_xlabel("position in sequence")
    ax.set_ylabel("mean attention weight")
    ax.set_title("Mean attention over positions by method")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig, "\n".join(summary_lines)


def _refresh(run: str, k: int):
    return _plot(run, k)


def _ks_for_run(run: str) -> list[int]:
    comp = _load_comparison(run)
    if comp is None:
        return [8]
    return comp.get("k_list", [8])


with gr.Blocks(title="attention_kth_select / first_pass") as demo:
    gr.Markdown(
        "# k-th position selection — hand-built positional head\n"
        "A single attention head with hand-set, position-only Q/K weights "
        "spikes at the addressed position **k**, while a content-matching head "
        "(attend to token 99) leaks onto spurious markers."
    )

    runs = _list_runs()
    default_run = runs[0] if runs else ""
    default_ks = _ks_for_run(default_run)

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(
                choices=runs, value=default_run, label="run", scale=2
            )
            k_dd = gr.Dropdown(
                choices=default_ks,
                value=8 if 8 in default_ks else (default_ks[0] if default_ks else 8),
                label="target position k", scale=1,
            )
        plot = gr.Plot(label="attention distribution")
        summary = gr.Textbox(label="per-method summary", lines=5)

        run_dd.change(
            lambda r: gr.update(choices=_ks_for_run(r),
                                value=8 if 8 in _ks_for_run(r) else _ks_for_run(r)[0]),
            inputs=run_dd, outputs=k_dd,
        )
        run_dd.change(_refresh, inputs=[run_dd, k_dd], outputs=[plot, summary])
        k_dd.change(_refresh, inputs=[run_dd, k_dd], outputs=[plot, summary])
        demo.load(_refresh, inputs=[run_dd, k_dd], outputs=[plot, summary])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
