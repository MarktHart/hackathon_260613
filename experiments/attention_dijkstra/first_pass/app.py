


import gradio as gr
from agentic.experiments import benchmark_panel, load_task

with gr.Blocks() as demo:
    gr.Markdown("# Attention Dijkstra – First Pass")
    gr.Markdown(
        "This attempt implements a simple attention-style mechanism that iteratively relaxes distances via soft-min aggregation (Bellman–Ford on the graph)."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            # Demo visualization: no interactive demo yet, just a placeholder.
            # In future attempts we can add a visualization of the graph and distance updates.
            gr.Markdown("Visualisation coming next!")
        with gr.Tab("Benchmark"):
            # Leaderboard and metric history across all attempts.
            benchmark_panel("experiments/attention_dijkstra")

if __name__ == "__main__":
    demo.launch()