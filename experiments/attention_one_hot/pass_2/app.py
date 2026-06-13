"""
Gradio Blocks app for `attention_one_hot`/`pass_2`.

Two tabs:
- **Demo**: interactive chart of headline metrics vs. sequence length.
  Select older runs from a dropdown; the plot updates.
- **Benchmark**: built from `agentic.experiments.benchmark_panel`; shows
  leaderboard, metric history, and per-run `one_hotness` per length.

The plot shows:
- **Peak mass**: average of the max attention weight per row, 1.0 = perfectly one-hot
- **Target mass**: mean attention placed on the true needle position
- **Argmax accuracy**: fraction of rows whose max is the correct target index
- **Uniform baseline (1/L)**: reference for no-attention strawman

All plots use the `sweep` records read from `bench.json`.
"""

from pathlib import Path
from typing import List, Dict

import pandas as pd
import plotly.graph_objects as go
import gradio as gr

from agentic.experiments import load_task, benchmark_panel, results_dirs, results_dir

# --------------------------------------------------------
# Helper: load a run's payload from disk
# --------------------------------------------------------


def load_payload(run_path: str) -> dict:
    bench_file = Path(run_path, "bench.json")
    if not bench_file.is_file():
        return {}
    try:
        data = json.load(bench_file.open())
    except Exception:
        return {}
    if not isinstance(data.get("sweep"), list):
        data["sweep"] = []
    return data


# --------------------------------------------------------
# Demo tab: render sweep curves
# --------------------------------------------------------


def _make_demo_df(bench: dict) -> pd.DataFrame:
    rows = []
    for rec in bench.get("sweep", []):
        rows.append({
            "length": int(rec["length"]),
            "peak_mass": float(rec["peak_mass"]),
            "target_mass": float(rec["target_mass"]),
            "selection_accuracy": float(rec["selection_accuracy"]),
            "uniform_peak_mass": float(rec["uniform_peak_mass"]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("length")


def _demo_plot(df: pd.DataFrame):
    fig = go.Figure()

    # Peak mass -> sharpness
    fig.add_trace(go.Scatter(
        x=df["length"], y=df["peak_mass"],
        mode="lines+markers+text",
        name="Peak attention mass (max row)",
        line=dict(color="#1f77b4"),
        marker=dict(size=6, color="#1f77b4"),
        text=[f"{p:.3f}" for p in df["peak_mass"]],
        textposition="bottom center",
        hovertemplate="<b>Peak=%{y}</b><br>Len=%{x}<extra></extra>"
    ))

    # Target mass (should track peak mass closely)
    fig.add_trace(go.Scatter(
        x=df["length"], y=df["target_mass"],
        mode="lines+markers",
        name="Mass on true needle",
        line=dict(color="#d62728"),
        marker=dict(size=6, color="#d62728")
    ))

    # Argmax accuracy
    fig.add_trace(go.Scatter(
        x=df["length"], y=df["selection_accuracy"],
        mode="lines+markers",
        name="Argmax accuracy",
        line=dict(color="#2ca02c"),
        marker=dict(size=6, color="#2ca02c")
    ))

    # Uniform reference (1/L) plotted as dashed grey
    unif_x = df["length"]
    unif_y = 1.0 / unif_x
    fig.add_trace(go.Scatter(
        x=unif_x, y=unif_y,
        mode="lines",
        name="Uniform baseline (1/L)",
        line=dict(color="grey", dash="dot")
    ))

    fig.update_layout(
        title="Attention one-hot behavior across sequence length sweep",
        xaxis=dict(title="Sequence length L", type="log", dtick=1),
        yaxis=dict(title="Attention mass / accuracy", range=[0.0, 1.05]),
        width=800,
        height=500,
        legend=dict(
            x=0.5, y=1.08, orientation="h",
            xanchor="center", yanchor="bottom"
        ),
        margin=dict(t=60, b=40, l=50, r=50),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig.to_json()


def demo_plot_callback(run_path: str) -> str:
    bench = load_payload(run_path)
    df = _make_demo_df(bench)
    if df.empty:
        return json.dumps({})
    return _demo_plot(df)


def demo_run_selector() -> List[str]:
    dirs = sorted(results_dirs(__file__), key=os.path.getmtime, reverse=True)
    return [d.name for d in dirs if d.is_dir()]


def demo_run_selector_callback(pick: str) -> str:
    run_path = Path(__file__).parent / pick
    return demo_plot_callback(run_path)


# --------------------------------------------------------
# Build Gradio app inside the demo block
# --------------------------------------------------------


with gr.Blocks() as demo:
    # Demo Tab
    with gr.Blocks():
        gr.Markdown(
            "# Attention One-Hot Demo: Learned Embedding Head (pass_2)\n\n"
            "Measures whether a single learned attention head produces a sharp one-hot attention pattern\n"
            "as sequence length L grows from 8 to 64."
        )
        with gr.Row():
            with gr.Column(scale=1):
                run_dd = gr.Dropdown(
                    label="Select run directory",
                    choices=demo_run_selector(),
                    value=demo_run_selector()[0] if demo_run_selector() else ""
                )
                run_dd.change(fn=demo_run_selector_callback, inputs=run_dd, outputs=plot)
            with gr.Column(scale=6):
                plot = gr.Plot(width=800, height=500)

    # Benchmark Tab — use the built-in benchmark_panel
    bench_panel = benchmark_panel(goal_dir="attention_one_hot")

    demo = gr.Blocks()

if __name__ == "__main__":
    demo.launch()