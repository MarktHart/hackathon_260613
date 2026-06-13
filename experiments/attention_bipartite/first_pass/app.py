import gradio as gr
from agentic.experiments import benchmark_panel, load_task, load_app_dir

from pathlib import Path

# Load the task and get the goal directory
task = load_task(__file__)
goal_dir = Path(task.__file__).parent

with gr.Blocks() as demo:
    with gr.Blocks() as demo:
        with gr.Tab("Demo"):
            gr.Markdown("Visual demo of bipartite attention mechanism")
            gr.Markdown("The model should focus on within-group attention (A->A and B->B) and ignore cross-group attention (A->B and B->A)")
        
        with gr.Tab("Benchmark"):
            benchmark_panel(goal_dir)

if __name__ == "__main__":
    demo.launch()