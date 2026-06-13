import gradio as gr
import json
import os
from agentic.experiments import benchmark_panel, results_dir

with gr.Blocks(theme="soft", title="Attention CYK Demo") as demo:
    # Demo tab for the specific mechanism's behavior
    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("# Attention CYK Demo")
                gr.Markdown("""
                This demo showcases how a hand-built attention mechanism identifies correct split points for CYK parsing.
                The mechanism assigns higher attention scores to bracket balance points (where depth returns to 0)
                and moderate scores to other potential split points.
                """)
            with gr.Column():
                # Example input
                gr.Label("Example bracket string:")
                example_seq = "(())()"
                example_seq_compact = " ".join(map(str, [0 if c == '(' else 1 for c in example_seq]))
                gr.Label(f"Input: {example_seq_compact}")
    
    # Benchmark tab
    with gr.Tab("Benchmark"):
        # Show benchmark panel for all attempts
        with gr.Row():
            with gr.Column():
                gr.Markdown("## Benchmark Results")
                gr.Markdown("View performance metrics across all attempts:")
                bench_panel = benchmark_panel(os.path.dirname(os.path.dirname(__file__)))
    
    # Load demo function
    demo.load(None, inputs=None, outputs=None)
    
if __name__ == "__main__":
    demo.launch()