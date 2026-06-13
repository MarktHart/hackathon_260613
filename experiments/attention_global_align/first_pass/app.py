import gradio as gr
from agentic.experiments import benchmark_panel

with gr.Blocks() as demo:
    benchmark_panel("experiments/attention_global_align")

if __name__ == "__main__":
    demo.launch()