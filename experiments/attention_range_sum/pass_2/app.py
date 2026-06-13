import gradio as gr
from agentic.experiments import benchmark_panel
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Load the latest run under results by default.
run_dir = Path(__file__).parent / "results" / sorted((Path(__file__).parent / "results").iterdir(), key=lambda p: p.name, reverse=True)[0]

def load_and_plot(run_dir: Path):
    # Parse the canonical sweep data from run_dir/benchmark.json
    # This demo only visualises the headline metrics: per-k MSE vs the linear baseline.
    benchmark_path = run_dir / "benchmark.json"
    import json
    data = json.load(benchmark_path.open("r"))
    sweep = data["sweep"]
    k_keys = sorted([k for k in data.keys() if k.startswith('range_sum_mse_k_')])
    k_vals = [int(k.split('_')[-1]) for k in k_keys]
    mse_vals = [float(data[k]) for k in k_keys]
    base_vals = [float(data[k.replace('range_sum', 'linear_baseline')]) for k in k_keys]

    plt.figure(figsize=(7, 4))
    plt.plot(k_vals, mse_vals, marker='o', label='model MSE', linewidth=2)
    plt.plot(k_vals, base_vals, marker='x', linestyle='--', label='no-mechanism MSE', linewidth=2, markersize=8)
    plt.title('MSE vs range length (small window = easier)', pad=10)
    plt.xlabel('Window size k')
    plt.ylabel('MSE')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xticks(k_vals)
    plt.legend()
    plt.tight_layout()
    img_path = run_dir / "demo_plot.png"
    plt.savefig(img_path, dpi=200)
    plt.close()
    return str(img_path), mse_vals, k_vals

def demo():
    # Demo tab: shows the plot and the latest model's name and seed.
    with gr.Blocks() as demo:
        gr.Markdown("# attention_range_sum Demo (pass_2)")
        gr.Markdown("This attempt trains a tiny single-head Transformer and loads it from `head.pth`. The head learns to broadcast a query that reads across the window [start, end) and to scale / sum the token values. The Demo tab visualises MSE on the canonical sweep. The app loads the most recent run by default; use the dropdown to compare older runs.")
        with gr.Blocks():
            img = gr.Image(label="MSE vs range length")
            info = gr.Markdown()
            plot_btn = gr.Button("Refresh plot")
            run_dd = gr.Dropdown(
                [d.name for d in sorted((Path(__file__).parent / "results").iterdir(), key=lambda p: p.name, reverse=True)],
                value=run_dir.name,
                label="Select run"
            )
            status = gr.Textbox(label="Status")

            def on_ddChange(selected_name):
                run_path = Path(__file__).parent / "results" / selected_name
                info.update(f"Loaded run: `{run_path.name}` (seed 42).\nHead parameters are fixed; no retraining in the UI.")
                return None, run_path, None

            run_dd.change(
                fn=on_ddChange,
                inputs=run_dd,
                outputs=[info, gr.State(), status]
            )

            def on_plotRefresh(run_dir_state):
                img_path = Path(__file__).parent / "results" / run_dir_state / "demo_plot.png"
                if not img_path.is_file():
                    # regenerate if missing
                    load_and_plot(run_dir_state)
                return str(img_path),  # image src
            plot_btn.click(
                fn=on_plotRefresh,
                inputs=run_dd,
                outputs=img
            )

        # Benchmark tab (static panel from agentic)
        with gr.Blocks():
            benchmark_panel(Path(__file__).parent.parent)  # scans all attempts in this goal

    return demo


if __name__ == "__main__":
    demo = demo()
    demo.launch()