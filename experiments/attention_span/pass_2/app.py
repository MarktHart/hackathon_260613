import gradio as gr
from agentic.experiments.benchmark_panel import leaderboard, metric_curve
from agentic.experiments import load_task

# Load the task to access the canonical sweep and metrics.
task = load_task(__file__)
payload = task.evaluate(task.random_model_fn())  # dummy run to extract keys.

with gr.Blocks() as demo:
    # Demo tab: raw sweep of mean attentions vs distance
    with gr.Blocks(variant="panel") as demo_tab:
        gr.Markdown("# Attention Span Decay Demo")
        gr.Markdown("""
        **Input**: a synthetic sequence where token_id `50256` (the *key*) sits at position 0, and
        all other positions are i.i.d. distractors from 1..1000.

        **Output**: a per-sequence `attention_to_key` array of shape `[1000, 1024]` reporting attention
        from each query position back to the key.

        This attempt uses a single attention head that applies a fixed exponential decay:
        `weight(d) = exp(-λ * d)`, with λ ≈ 0.02. The line plot below shows per-distance mean attention.
        """)

        x = np.array(payload["distances"])
        y = np.array([rec["mean_attention"] for rec in payload["sweep"]])
        gr.Plot(
            x, y, title="Mean Attention vs Distance",
            x_label="distance from key ( tokens )", y_label="mean attention", width=800, height=500
        )

    # Benchmark tab: reuse the shared leaderboard/curve panel
    with gr.Blocks(variant="panel") as benchmark_tab:
        gr.Markdown("# Benchmark Leaderboard")
        gr.Markdown("The table below ranks attempts by metrics from `benchmark.py`. Use this tab to compare your method against the field.")
        leaderboard(__file__)

        gr.Markdown("#### Metric trend over runs")
        metric_curve(__file__)

    # Two tab layout
    gr.Tabs(
        items=[
            gr.TabsItem(label="Demo", component=demo_tab),
            gr.TabsItem(label="Benchmark", component=benchmark_tab),
        ]
    )

if __name__ == "__main__":
    demo.launch()