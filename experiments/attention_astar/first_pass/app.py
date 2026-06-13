import gradio as gr
from agentic.experiments import benchmark_panel, load_task

with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention A* Demo (first_pass)\n"
        "Hand-coded attention head implementing the A* heuristic: f = g + h"
    )
    with gr.Tab("Visualization"):
        with gr.Blocks():
            with gr.Row():
                gr.Markdown("### 2D Grid (8×8, 20% obstacles)\n"
                          "The attention head evaluates a 3×3 window around the agent\n"
                          "Logits = center_offset (g) + manhattan_to_goal (h)\n"
                          "Obstacles return -inf, center position masked out")
                with gr.Blocks():
                    pass
            with gr.Row():
                gr.Button("No demo interactivity needed — hand-built circuit is deterministic")
    demo.load(lambda: None, None, None)  # ensure blocks are initalised

    with gr.Tab("Benchmark"):
        benchmark_panel.load_panel(demo, load_task(__file__))

if __name__ == "__main__":
    demo.launch()