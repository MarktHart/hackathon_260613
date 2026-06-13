import gradio as gr

from agentic.experiments import load_task, benchmark_panel
from experiments.attention_count.task import generate, evaluate, random_model_fn
from experiments.attention_count.benchmark import score


# ---------------------------------------------------------------------------
# 1. Demo tab: the visualisation we care about
# ---------------------------------------------------------------------------
def demo_tab() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("## Attention Count Demo")
        gr.Markdown("The model attempts to recover the true number of matching positions in contexts of fixed length.")

        # The demo can only show a slice; we chose a slice-by-slice chart.
        # For each true count m, show bar pairs: model mae vs baseline mae.
        def make_demo_plot() -> gr.Plot:
            # Re-generate the canonical batch (seed 0).
            batch = generate(SEED=0)
            mae_model = []
            mae_baseline = []
            true_counts = list(batch.counts)
            for m, (k, v) in zip(true_counts, batch.sweep):
                baseline = [b for b in batch.baseline_sweep if b["true_count"] == m][0]
                mae_model.append(v["mae"])
                mae_baseline.append(baseline["mae"])
            # Simple bar chart: two bars per m.
            import numpy as np, matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(9, 4))
            ind = np.arange(len(true_counts))
            width = 0.35
            bars1 = ax.bar(ind - width/2, mae_model, width, label="model", color="#4a90e2")
            bars2 = ax.bar(ind + width/2, mae_baseline, width, label="constant baseline", color="#a0a0a0")
            ax.set_xlabel("true count m")
            ax.set_ylabel("MAE")
            ax.set_title("Per-slice MAE vs constant predictor")
            ax.set_ylim(0, 1.0)
            ax.set_xticks(ind)
            ax.set_xticklabels(true_counts)
            ax.legend()
            fig.tight_layout()
            return fig

        bar_plot = gr.Plot(label="MAE per true count", value=None, height=300)
        demo.load(fn=make_demo_plot, outputs=bar_plot)

        # Optional table of payload keys; just a quick data inspection.
        with gr.Blocks() as payload_panel:
            gr.Markdown("### Payload preview (first few records, JSON-like)")
            # For brevity we just show a subset.
            def show_payload():
                batch = generate(SEED=0)
                payload = evaluate(random_model_fn())
                # Simulate our model's payload.
                actual_payload = evaluate(lambda q, k, v: np.mean(k @ q))   # dummy
                keys = sorted(actual_payload.keys())
               short = {k: str(actual_payload[k]) for k in keys[:8]}
                return gr.JSON(short)
            payload_table = gr.JSON(label="Payload snapshot")
            demo.load(fn=show_payload, outputs=payload_table)

    return demo


# ---------------------------------------------------------------------------
# 2. Benchmark tab: shared across all attempts in this goal
# ---------------------------------------------------------------------------
def benchmark_tab() -> gr.Blocks:
    with gr.Blocks() as bm:
        # Use the shared benchmark_panel (scans the whole goal directory)
        benchmark_panel(bm, "attention_count")
    return bm


# ---------------------------------------------------------------------------
# 3. Gradio entrypoint
# ---------------------------------------------------------------------------
def demo() -> gr.Blocks:
    with gr.Blocks() as full_app:
        with gr.Tabs():
            with gr.Tab("Demo"):
                demo_tab()
            with gr.Tab("Benchmark"):
                benchmark_tab()
    return full_app


# ---------------------------------------------------------------------------
# Boot-check export: module-level demo
# ---------------------------------------------------------------------------
demo: gr.Blocks = demo()

if __name__ == "__main__":
    demo.launch()