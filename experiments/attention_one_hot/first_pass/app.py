"""
Gradio app for `attention_one_hot/first_pass`:

Demo tab: interactive plot of one-hot accuracy / mass_on_correct / entropy vs. margin.

Benchmark tab: `agentic.experiments.benchmark_panel` showing leaderboard and metric history
across this goal and runs.

All plots use the per-sweep record keys defined in `payload["sweep"]`:
* "margin": float (x-axis)
* "one_hot_accuracy": float
* "mass_on_correct": float
* "attn_entropy": float

We read the latest run by default; a dropdown lets the user select older runs under the same attempt.

Style is minimal — the goal is to let the numbers speak.
"""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import gradio as gr

from agentic.experiments import (load_task, benchmark_panel, results_dirs,
                                 results_dir)

# --------------------------------------------------------
# Utility to load a `benchmark.json` from a results directory
# --------------------------------------------------------


def load_payload(path: str) -> dict:
    """Reads `bench.json` inside `path`. Returns dict; never raises."""
    bench_path = os.path.join(path, "bench.json")
    if not os.path.isfile(bench_path):
        return {}
    try:
        with open(bench_path, "r") as f:
            data = json.load(f)
    except Exception:
        return {}
    # The key we care about for gradio is 'sweep'
    if not isinstance(data.get("sweep"), list):
        data["sweep"] = []
    if not isinstance(data.get("version"), int):
        data["version"] = None
    return data


# --------------------------------------------------------
# Demo tab: plot sweep curves
# --------------------------------------------------------


def _fmt_margin(m: float) -> str:
    # used by the API keys, not the demo UI
    return f"{m:.1f}".replace(".", "p")


def _make_demo_df(bench: dict) -> pd.DataFrame:
    rows = []
    for rec in bench.get("sweep", []):
        rows.append({
            "margin": float(rec["margin"]),
            "one_hot_accuracy": float(rec["one_hot_accuracy"]),
            "mass_on_correct": float(rec["mass_on_correct"]),
            "attn_entropy": float(rec["attn_entropy"]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("margin")


def _demo_plot(df: pd.DataFrame):
    """Rendered with Plotly; returns JSON serializable."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Accuracy plot
    fig.add_trace(go.Scatter(
        x=df["margin"], y=df["one_hot_accuracy"],
        mode="lines+markers", name="Argmax accuracy",
        line=dict(color="#1f77b4"), marker=dict(size=6, color="#1f77b4")
    ))

    # Mass-on-correct plot (second y-axis for clarity)
    fig.add_trace(go.Scatter(
        x=df["margin"], y=df["mass_on_correct"] * 1.2,  # shift slightly for readability
        mode="lines+markers", name="Mass on correct key",
        line=dict(color="#d62728"), marker=dict(size=6, color="#d62728")
    ))

    # Entropy (third y-axis, inverted)
    fig.add_trace(go.Scatter(
        x=df["margin"], y=-df["attn_entropy"],
        mode="lines+markers", name="Negative entropy (sharper = lower)",
        line=dict(color="#2ca02c"), marker=dict(size=6, color="#2ca02c")
    ))

    fig.update_layout(
        title="Attention one-hot behavior across margin sweep",
        xaxis_title="Injected margin (logit strength)",
        yaxis=dict(title="Argmax accuracy"),  # primary y
        yaxis2=dict(
            title="Mean attention mass on correct key",
            overlaying="y",
            side="right",
            range=[0.0, 1.0],
            position=1.10  # shift right of primary
        ),
        yaxis3=dict(
            title="Negative Shannon entropy (nats)",
            overlaying="y",
            side="right",
            position=1.18,
            range=[-12.0, 0.0]
        ),
        width=800,
        height=500,
        legend=dict(x=0.5, y=1.15, orientation="h", xanchor="center", yanchor="bottom")
    )
    return fig.to_json()


def demo_plot_callback(run_path: str, _: gr.Slider) -> str:
    bench = load_payload(run_path)
    df = _make_demo_df(bench)
    if df.empty:
        return json.dumps({}, default=str)
    return _demo_plot(df)


def demo_run_selector() -> List[str]:
    return sorted(results_dirs(__file__), key=os.path.getmtime, reverse=True)


def demo_run_selector_callback(pick: str) -> str:
    return demo_plot_callback(pick, 1.0)  # dummy slider value


# --------------------------------------------------------
# Build the Gradio app
# --------------------------------------------------------


with gr.Blocks() as demo:
    # ----------------------------------------------------------------
    # Demo Tab
    # ----------------------------------------------------------------
    with gr.Blocks():
        gr.Markdown(
            "# Attention One-Hot Demo (first_pass)\n\n"
            "Measures whether a single scaled-dot-product attention head produces a sharp one-hot attention pattern\n"
            "as the query→key signal margin varies."
        )

        with gr.Row():
            with gr.Column(scale=1):
                run_dd = gr.Dropdown(
                    label="Select run directory",
                    choices=demo_run_selector(),
                    value=demo_run_selector()[0]
                )
                btn = gr.Button("Refresh plot")
            with gr.Column(scale=6):
                plot = gr.Plotlabel="Sweep curves: accuracy / mass / entropy")
                s = gr.Slider(
                    label="Interaction slider (demo placeholder)", value=1.0, visible=False
                )

        run_dd.change(fn=demo_run_selector_callback, inputs=run_dd, outputs=plot)
        btn.click(fn=demo_plot_callback, inputs=[run_dd, s], outputs=plot)

    # ----------------------------------------------------------------
    # Benchmark Tab
    # ----------------------------------------------------------------
    bench_panel = benchmark_panel(goal_dir="attention_one_hot")

    demo = gr.Blocks()

if __name__ == "__main__":
    demo.launch()