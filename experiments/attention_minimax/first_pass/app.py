import gradio as gr
from agentic.experiments import benchmark_panel

with gr.Blocks() as demo:
    with gr.Tab("Demo"):
        gr.Markdown("# attention_minimax")
        # Demo panel could show per-alpha attention weights, visual comparison with uniform baseline, etc.
        gr.Markdown("Placeholder for the visualisation of attention weights over the canonical sweep (alpha from 0.0 to 1.0).")
    tab = benchmark_panel("experiments/attention_minimax")

if __name__ == "__main__":
    demo.launch()