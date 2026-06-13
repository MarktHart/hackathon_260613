"""Gradio: per-token attention mass, softmax vs linear baseline, with a scale slider."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _runs() -> list[str]:
    if not RESULTS.exists():
        return []
    return sorted((p.name for p in RESULTS.iterdir() if p.is_dir()), reverse=True)


def _load(run_id: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / run_id / "weights.csv")


def _nearest_scale(df: pd.DataFrame, scale: float) -> float:
    available = sorted(df["scale"].unique())
    return float(min(available, key=lambda s: abs(s - scale)))


def render(run_id: str, scale: float) -> Figure:
    df = _load(run_id)
    s = _nearest_scale(df, scale)
    sub = df[df["scale"] == s].reset_index(drop=True)
    tokens = sub["token"].tolist()
    x = np.arange(len(tokens))
    width = 0.4

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, sub["softmax_weight"], width=width, label="softmax", color="#d62728")
    ax.bar(
        x + width / 2,
        sub["linear_weight"],
        width=width,
        label="linear baseline",
        color="#1f77b4",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(tokens, rotation=15, ha="right")
    ax.set_ylabel("attention mass")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"per-token attention at scale = {s}")
    ax.axhline(1.0 / len(tokens), color="gray", linestyle="--", linewidth=0.8, label="uniform")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def build() -> gr.Blocks:
    runs = _runs()
    default_run = runs[0] if runs else None

    with gr.Blocks(title="Attention as a soft AND") as demo:
        gr.Markdown(
            "# Attention as a soft AND\n"
            "Same per-token scores, two normalisations. Softmax exponentiates first → the "
            "score for the `both` token (sum of two positive matches) blows up multiplicatively "
            "and grabs almost all the mass. The linear baseline (no `exp`) stays diffuse. "
            "Slide *scale* up to see the AND-spike sharpen."
        )
        with gr.Row():
            run_dd = gr.Dropdown(choices=runs, value=default_run, label="Run", interactive=True)
            scale_slider = gr.Slider(minimum=0.25, maximum=3.0, value=1.0, step=0.25, label="Scale")
        plot = gr.Plot()

        inputs = [run_dd, scale_slider]
        for ev in (run_dd.change, scale_slider.change):
            ev(render, inputs=inputs, outputs=plot)
        demo.load(render, inputs=inputs, outputs=plot)

    return demo  # type: ignore[no-any-return]


demo = build()


if __name__ == "__main__":
    demo.launch()
