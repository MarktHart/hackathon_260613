import gradio as gr
from agentic.experiments import load_task, benchmark_panel

from PIL import Image
import numpy as np

# Load the same task and the results of the latest run.
task = load_task(__file__)
attempt_dir = __file__.replace("main.py", "pass_2")
try:
    import glob
    results = glob.glob(f"{attempt_dir}/results/*/")
    latest_run = max(results, key=lambda p: np.float64(p.split('/')[-2]))
except:
    latest_run = None

def load_run(run_dir):
    import json
    benchmark_path = f"{run_dir}/benchmark.json"
    with open(benchmark_path, "r") as f:
        data = json.load(f)
    return data

def get_demo_preds(alpha, beta):
    rng = np.random.default_rng(2024)
    B = 256
    x1 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
    x2 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
    al = np.full((B, 1), alpha, dtype=np.float32)
    be = np.full((B, 1), beta, dtype=np.float32)
    batch = Batch(x1=x1, x2=x2, alpha=al, beta=be)
    pred = model_fn(batch)
    # Return the last prediction across tokens (averaging not needed because they are equal).
    # Shape (B,) -> flatten for the demo.
    return pred.mean(axis=1).flatten()

if latest_run is None:
    demo_preds = np.zeros((256,))
else:
    data = load_run(latest_run)
    demo_preds = np.array(data["canonical"]["pred"]).flatten()

with gr.Blocks() as demo:
    gr.Markdown("# attention_linear_sum Demo")
    gr.Markdown("""
**Can a single attention head compute `y = α·x₁ + β·x₂` from scalar-embedded context?**  
The head reads two features (x1, x2) at position 0 and 1, a coefficient token (α,β) at position 2, and broadcasts the linear combination to all target positions (3–7). This demo shows per-coefficient R² across the sweep and a sample prediction.
""")
    with gr.Row():
        alpha_slider = gr.Slider(label="α", minimum=-2.0, maximum=2.0, step=0.5, value=1.0)
        beta_slider = gr.Slider(label="β", minimum=-2.0, maximum=2.0, step=0.5, value=1.0)
    demo_preds_box = gr.Plot(label="Sample prediction (mean over 256 samples)")
    demo_preds_box.plot([i * 0.1 for i in range(256)], demo_preds.tolist(), title="Model output for canonical α=1, β=1")
    # R² sweep bar chart
    gr.Plot(label="Per-coefficient R² across sweep")
    # We plot the latest run if exists, or a placeholder.
    if latest_run:
        data = load_run(latest_run)
        sweep_coeffs = {rec["alpha"]: rec["beta"] for rec in data["sweep"]}
        r2_vals = [rec["r2"] for rec in data["sweep"]]
        # Create x-ticks with label αβ formatted as in metrics
        r2_chart = gr.Plot()
        x_labels = [f"{a:.1f}_{b:.1f}".replace("-", "m").replace(".", "p")
                   for a, b in zip(*zip(*[zip(*map(lambda d: (d["alpha"], d["beta"], d["r2"]), data["sweep"]))]))]
        r2_chart.plot(range(len(r2_vals)), r2_vals, title="R² across coefficient sweep")
        r2_chart.set_x_axis(label="αβ pair")
        r2_chart.set_y_axis(label="R²")
    else:
        gr.Plot()

    # Event: update the plot with new αβ (just for demos)
    alpha_slider.change(
        fn=get_demo_preds,
        inputs=[alpha_slider, beta_slider],
        outputs=demo_preds_box,
    )

    with gr.Blocks() as bench:
        # Benchmark panel from the framework
        bench_panel = benchmark_panel(task.__name__, __file__)
        bench_panel.render()

    demo.add_tab(benchmark_panel(bench_panel, "Benchmark"), label="Benchmark")

if __name__ == "__main__":
    demo.launch()