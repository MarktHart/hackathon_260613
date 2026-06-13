import gradio as gr
import numpy as np
from agentic.experiments import load_task, benchmark_panel

def build_plot(cos_AB_val, run_path="results"):
    """"Return bar chart of mean scores per condition for a given cos_AB slice."""
    batch = load_task(__file__).generate()
    #Find the batch corresponding to the selected cos_AB
    for rec in batch["sweep_raw"]:
        if np.isclose(rec["cos_AB"], cos_AB_val):
            q = rec["q_vectors"]
            k = rec["k"]
            break
    #Run the model function on that batch slice
    def prod_head(q, k):
        w_q = np.eye(64)
        w_k = np.eye(64) * 0.9
        scale = np.sqrt(64)
        w_q_proj = q @ w_q
        w_k_proj = k @ w_k
        return (w_q_proj * w_k_proj) / scale
    scores = prod_head(q, k)

    #Aggregate mean scores per condition
    labels = ["(0,0)", "(1,0)", "(0,1)", "(1,1)"]
    means = np.zeros(4)
    for i, (a, b) in enumerate([[0,0], [1,0], [0,1], [1,1]]):
        mask = (np.isclose(batch["conditions"], [a, b]).all(axis=1))
        means[i] = scores[mask].mean()

    #Return a plain numpy array for the chart — Gradio expects an array of shape (4,)
    return means

with gr.Blocks(theme=gr.Themes.default()) as demo:
    gr.Markdown("# Attention AND (product head model)")
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown("Bar chart of mean attention scores for the four presence conditions, across feature interference level (cos AB).")
            cos_slider = gr.Slider(minimum=0.0, maximum=0.9, step=0.1, label="Cos(e_A, e_B)", value=0.0)
            with gr.Row():
                chart = gr.BarPlot(
                    title="Mean attention score per condition",
                    y_label="Score",
                    x_title="Condition (α_A, α_B)",
                    width=600,
                    height=300,
                )
            demo.load(fn=lambda: build_plot(0.0), outputs=chart)
            cos_slider.change(
                fn=build_plot,
                inputs=cos_slider,
                outputs=chart
            )
        with gr.TabItem("Benchmark"):
            benchmark_panel(__file__)

if __name__ == "__main__":
    demo.launch()