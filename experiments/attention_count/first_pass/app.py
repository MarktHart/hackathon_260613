import gradio as gr
from agentic.experiments import benchmark_panel

goal_dir = "experiments/attention_count"

with gr.Blocks() as demo:
    gr.Markdown("# attention_count — first_pass")
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown(
                "This attempt uses a softmax attention readout with a heuristic "
                "calibration. The Demo tab shows the per-slice accuracy and MAE."
            )
            # The benchmark panel already shows sweep metrics; we just surface it here too.
            benchmark_panel(goal_dir).render()
        with gr.TabItem("Benchmark"):
            benchmark_panel(goal_dir).render()

if __name__ == "__main__":
    demo.launch()