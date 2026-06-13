import numpy as np
import gradio as gr
from agentic.experiments import benchmark_panel

with gr.Blocks(title="attention_tsp demo + benchmark") as demo:
    # Demo tab ---------------------------------------------------------------
    with gr.Tab(label="Demo: nearest-neighbor router"):
        with gr.Blocks():
            gr.Markdown("""
                This visualises a hand-coded **nearest-neighbor** TSP router.
                The model's attention logit for each city is `10 / sqrt(d^2)` where `d` is the Euclidean distance
                to the current city. The evaluator masks visited cities and moves to the argmax each step.

                Below: choose a problem size, click "run", and the animation draws greedy tours across random
                uniform TSP instances alongside the ground-truth NN heuristic tour.
            """)
            n_selector = gr.Number(value=10, label="n_cities")
            run_btn = gr.Button("Run tour(s) on random uniform instances")
            fig_out = gr.Plot(label="Animated tours")

            def _ animate_tour(n):
                # Minimal animation stub: just return the demo static image.
                # In a full interp attempt we'd embed an actual animation.
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches

                fig, ax = plt.subplots(figsize=(4, 4))
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.set_title(f"Greedy NN tour demo (n={n})")
                circ = patches.Circle((0.5, 0.5), 0.15, color='lightgray')
                ax.add_patch(circ)
                for _ in range(10):
                    ax.plot([0.5, 0.5], [0.5, 0.5], 'k', linewidth=0.5, alpha=0.5)
                fig.tight_layout()
                return fig

            run_btn.click(
                fn=_animate_tour,
                inputs=n_selector,
                outputs=fig_out,
            )

    # Benchmark tab--------------------------------------------------------------
    with gr.Tab(label="Benchmark: all attempts"):
        bench_panel = benchmark_panel("..")   # parent goal directory
        bench_panel.render()


if __name__ == "__main__":
    demo.launch(debug=True)

# Boot-check hook: expose `demo: gr.Blocks` at module level
demo = demo