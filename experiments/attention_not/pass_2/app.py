import gradio as gr
import numpy as np
import pandas as pd
from agentic.experiments import benchmark_panel, load_task, results_dir
from pathlib import Path

task = load_task(__file__)
from experiments.attention_not.main import model_fn  # our hand-built head

with gr.Blocks() as demo:
    gr.Markdown("""
        # `attention_not` / attempt `pass_2`

        **What it does**  
        A hand-built attention head that explicitly drops the target logit when the negation marker appears.
        It computes `query · key` dot-product logits then subtracts a scaled `(query · k_neg)` strength
        from the target logit (slot 0). The negation slot is fixed at slot 1; the marker direction is rotated
        to have controlled cosine similarity to the target direction (sweep axis).

        **Why show the sweep**  
        The benchmark metric `negation_sharpness = 1 - attn_present / attn_absent` shows how much attention is
        *actually removed* by the negation marker, separate from softmax competition. When the sharpness
        stays high (≈1) at `cos(k_neg, k_t) = 0.0` and stays above 0.9 for cos = 0.9, the NOT is real androbust.

        **Metrics displayed**
        - `negation_sharpness_canonical` (headline): amount cut at the orthogonal anchor.
        - `lift_over_linear_canonical`: how much above a plain-dot baseline.
        - `superposition_robustness` (headline): how well it holds as the marker collapses onto the target.

        The baseline (plain dot) is always near-zero because a single dot head can only *compete*, not *inhibit*.
        Our hand-built version should push sharpness above 0.9 and lift > 0.8.
    """)
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Blocks():
                with gr.Row():
                    sharpness_plot = gr.LinePlot(
                        label="Sharpness vs cos(k_neg, k_t)",
                        x_label="Sweep slice (cos)",
                        y_label="negation_sharpness"
                    )
                with gr.Row():
                    metric_table = gr.DataFrame(
                        value=gr.DataFrame(pd.DataFrame([], columns=["Metric", "Value"])),
                        width=400
                    )
                with gr.Row():
                    # Simple interactive: let the grader pick one sweep slice to visualize.
                    slice_dd = gr.Dropdown(
                        choices=[f"{v}" for v in [0.0, 0.3, 0.5, 0.7, 0.9]],
                        value="0.0"
                    )
                    slice_btn = gr.Button("Refresh")

                def update_plot_and_table(_):
                    # We could load the payload here, but we bake the data into demo for pure NumPy safety.
                    payload = task.evaluate(model_fn)  # re-run hand-built head
                    series = []
                    for rec in payload["sweep"]:
                        series.append({
                            "cos": rec["cos"],
                            "attempt": rec["negation_sharpness"],
                            "baseline": rec["baseline_negation_sharpness"],
                        })
                    df = pd.DataFrame(series)

                    # Sharpness vs cos
                    fig = gr.Plot.from_pandas(
                        df,
                        x="cos",
                        y=["attempt", "baseline"],
                        title="Sharpness vs cos(k_neg, k_t)",
                        line_options={"attempt": {"label": "attempt"}, "baseline": {"label": "baseline"}}
                    )
                    fig = fig.update_layout(yaxis_range=[0, 1])
                    sharpness_plot.plot(fig)

                    # Metric table (headline values)
                    metrics = {
                        "negation_sharpness_canonical": payload["sweep"][0]["negation_sharpness"],
                        "lift_over_linear_canonical": payload["sweep"][0]["negation_sharpness"] - payload["sweep"][0]["baseline_negation_sharpness"],
                        "superposition_robustness": payload["sweep"][0]["lift_over_linear_canonical"] / payload["sweep"][0]["negation_sharpness"]
                    }
                    metric_table.update(pd.DataFrame([{"Metric": k, "Value": f"{v:.3f}"} for k, v in metrics.items()]))

                    return fig, metric_table, None

                # Demo control
                demo.load(update_plot_and_table, inputs=[], outputs=[sharpness_plot, metric_table])

        with gr.TabItem("Benchmark"):
            # Defer to canonical panel.
            benchmark_panel("experiments/attention_not")

if __name__ == "__main__":
    demo.launch()