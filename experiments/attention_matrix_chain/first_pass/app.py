import gradio as gr
from agentic.experiments import benchmark_panel

with gr.Blocks() as demo:
    gr.Markdown("# Attention Matrix Chain Demo")
    gr.Markdown(
        "This is a baseline first-pass attempt: a model that _only_"
        " computes the direct matrix product \\(A_\\text{chain} = A_2 @ A_1\\)."
        " The goal is to measure how robust that composition is as"
        " attention rows become more peaked (`alpha` decreases)."
    )

    with gr.Blocks() as demo_tab:
        gr.Markdown("### Controls (no runtime variables required)")
        # Placeholder demo UI; the real story lives in the Benchmark tab
        info = gr.Markdown("All computation is fixed: GPU compute of `A2 @ A1`")
        info

    demo_tab

    with gr.Blocks() as benchmark_tab:
        benchmark_panel("../..")

    demo_tab, benchmark_tab

if __name__ == "__main__":
    demo.launch()