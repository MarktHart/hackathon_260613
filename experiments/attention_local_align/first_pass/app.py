import gradio as gr
from agentic.experiments import benchmark_panel
from pathlib import Path

GOAL_DIR = Path(__file__).parent.parent

with gr.Blocks() as demo:
    gr.Markdown("# attention_local_align — first_pass")
    gr.Markdown("Hand-built predecessor attention (shift = -1).")

    with gr.Tab("Demo"):
        gr.Markdown("""
        This attempt uses a **hand-built attention pattern** where every token attends
        exclusively to its immediate predecessor (index `t-1`). The pattern is a strict
        sub-diagonal band of width 1.

        Since the synthetic data generator *also* uses shift = -1 as the canonical
        condition, this should achieve near-perfect alignment on that shift and near-zero
        on all others.
        """)
        with gr.Row():
            shift_slider = gr.Slider(-2, 2, value=-1, step=1, label="Shift (ground-truth target)")
            metric_dd = gr.Dropdown(
                ["mean_max_attn_to_target", "mean_entropy", "frac_peak_on_target"],
                value="mean_max_attn_to_target",
                label="Metric"
            )
        out_plot = gr.Plot(label="Sweep across shifts")

        def plot_sweep(shift_val, metric_name):
            import json
            import matplotlib.pyplot as plt
            import numpy as np

            # Load latest run
            results_dir = Path(__file__).parent / "results"
            runs = sorted(results_dir.glob("*/benchmark.json"))
            if not runs:
                fig, ax = plt.subplots()
                ax.text(0.5, 0.5, "No runs yet", ha="center", va="center")
                return fig
            latest = runs[-1]
            with open(latest) as f:
                bm = json.load(f)

            payload = bm.get("payload", {})
            sweep = payload.get("sweep", [])

            shifts = [s["shift"] for s in sweep]
            vals = [s[metric_name] for s in sweep]

            fig, ax = plt.subplots(figsize=(6, 3))
            ax.bar([str(s) for s in shifts], vals, color=["#1f77b4" if s == shift_val else "#aec7e8" for s in shifts])
            ax.set_xlabel("Shift")
            ax.set_ylabel(metric_name)
            ax.set_title(f"Attention alignment sweep — {metric_name}")
            ax.axhline(y=1/31, color="gray", linestyle="--", label="Uniform baseline (1/(T-1))")
            ax.axhline(y=1/32, color="red", linestyle=":", label="Random peak baseline (1/T)")
            ax.legend(fontsize=8)
            return fig

        shift_slider.change(plot_sweep, inputs=[shift_slider, metric_dd], outputs=out_plot)
        metric_dd.change(plot_sweep, inputs=[shift_slider, metric_dd], outputs=out_plot)
        demo.load(plot_sweep, inputs=[shift_slider, metric_dd], outputs=out_plot)

    with gr.Tab("Benchmark"):
        # Drop-in panel that scans all attempts under the goal
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()