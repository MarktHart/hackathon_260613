from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Dict
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from agentic.experiments import (
    load_task,
    record_benchmark,
    results_dirs,
    results_dir,
    benchmark_panel,
)
import gradio as gr

# ----------------------------------------------------------------------
# Demo tab: visualize attention mass across sequence lengths for a chosen run
# ----------------------------------------------------------------------
def load_payload(run_path: str) -> dict:
    bench = Path(run_path) / "bench.json"
    if not bench.is_file():
        return {}
    try:
        return json.loads(bench.read_text())
    except Exception:
        return {}


def make_demo_df(bench: dict) -> pd.DataFrame:
    rows = []
    for rec in bench.get("sweep", []):
        rows.append({
            "length": int(rec["length"]),
            "peak_attn": float(rec["peak_attention"]),
            "target_attn": float(rec["target_attention"]),
            "entropy": float(rec["attention_entropy"]),
            "output_cosine": float(rec["output_cosine"]),
            "uniform_baseline": 1.0 / int(rec["length"]),
        })
    return pd.DataFrame(rows).sort_values("length")


def plot_demo_metrics(df: pd.DataFrame):
    fig = go.Figure()

    # Peak attention (should be close to 1 for one-hot)
    fig.add_trace(go.Scatter(
        x=df["length"], y=df["peak_attn"],
        mode="lines+markers",
        name="Peak attention mass",
        line=dict(color="#1f77b4"),
        marker=dict(size=6, color="#1f77b4")
    ))

    # Target attention on the correct needle position
    fig.add_trace(go.Scatter(
        x=df["length"], y=df["target_attn"],
        mode="lines+markers",
        name="Mass on correct needle",
        line=dict(color="#d62728"),
        marker=dict(size=6, color="#d62728")
    ))

    # Argmax accuracy (if we tracked it, we could add)
    # For now, we rely on peak_attn ≈ target_attn to imply correct argmax.

    # Entropy of the distribution
    fig.add_trace(go.Scatter(
        x=df["length"], y=df["entropy"],
        mode="lines+markers",
        name="Attention entropy (nats)",
        line=dict(color="#2ca02c"),
        marker=dict(size=6, color="#2ca02c")
    ))

    # Uniform baseline 1/L (dashed grey)
    unif_x = df["length"]
    unif_y = 1.0 / unif_x
    fig.add_trace(go.Scatter(
        x=unif_x, y=unif_y,
        mode="lines",
        name="Uniform baseline 1/L",
        line=dict(color="grey", dash="dot")
    ))

    fig.update_layout(
        title="One-hot Attention Behavior Across Sequence Length",
        xaxis_title="Length L (log)",
        yaxis_title="Attention Mass / Entropy",
        xaxis_type="log",
        width=800,
        height=500,
        legend=dict(
            x=0.5, y=1.08, orientation="h",
            xanchor="center", yanchor="bottom",
            bgcolor="white"
        )
    )
    return fig.to_json()


def demo_run_selector() -> List[str]:
    return sorted([d.name for d in results_dirs(__file__) if d.is_dir()],
                  key=os.path.getmtime, reverse=True)


def demo_plot_callback(pick: str) -> str:
    run_path = Path(__file__).parent / pick
    bench = load_payload(run_path)
    df = make_demo_df(bench)
    if df.empty:
        empty_fig = go.Figure().to_json()
        return empty_fig
    return plot_demo_metrics(df)


# ----------------------------------------------------------------------
# Build Gradio app with Demo and Benchmark tabs
# ----------------------------------------------------------------------
with gr.Blocks() as demo:
    with gr.Blocks():
        gr.Markdown(
            "# Attention One-Hot Demo — pass_3\n\n"
            "A small trainable attention head fitted to select exactly the key that matches\n"
            "the query. The goal is a sharp one-hot pattern (`≈1` on the needle, `≈0` elsewhere).\n"
            "- Plot shows peak mass, target mass, entropy, and uniform baseline across\n  lengths `L=16–256`."
        )

        with gr.Row():
            with gr.Column(scale=1):
                run_dd = gr.Dropdown(
                    choices=demo_run_selector(),
                    label="Select Run",
                    value=demo_run_selector()[0] if demo_run_selector() else ""
                )
                run_dd.change(fn=demo_plot_callback, inputs=run_dd, outputs=plot_out)
            with gr.Column(scale=6):
                plot_out = gr.Plot(width=800, height=500)
                plot_out.change(fn=demo_plot_callback, inputs=plot_out)
                

    # Benchmark tab: built-in benchmark_panel
    bench_panel = benchmark_panel(goal_dir="attention_one_hot")


if __name__ == "__main__":
    demo.launch()