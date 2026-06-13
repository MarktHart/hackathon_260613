import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from agentic.experiments import benchmark_panel, results_dir, load_task

# Load the task to access generate/evaluate for demo visualizations
task = load_task(__file__)

# Hand-coded model for demo (same as main.py but with numpy for plotting)
def demo_model_fn(coords, current_idx, visited):
    n = coords.shape[0]
    current = coords[current_idx:current_idx+1]
    diff = current - coords
    sqdist = (diff * diff).sum(axis=-1)
    logits = -sqdist * 10.0
    logits[visited] = -1e9
    return logits

def run_demo_tour(n_cities: int, seed: int = 123):
    """Generate and plot a tour for a single instance."""
    # Generate instance
    batch = task.generate(seed=seed)
    # Find first instance with matching n
    idx = next(i for i, n in enumerate(batch.ns) if n == n_cities)
    coords = batch.coords_list[idx]
    n = coords.shape[0]
    
    # Run greedy decode
    visited = np.zeros(n, dtype=bool)
    visited[0] = True
    current = 0
    order = [0]
    
    for _ in range(n - 1):
        logits = demo_model_fn(coords, current, visited)
        choice = int(np.argmax(logits))
        order.append(choice)
        visited[choice] = True
        current = choice
    
    # Also compute true NN tour for comparison
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt((diff ** 2).sum(axis=-1))
    
    nn_visited = np.zeros(n, dtype=bool)
    nn_visited[0] = True
    nn_current = 0
    nn_order = [0]
    for _ in range(n - 1):
        masked = np.where(nn_visited, np.inf, dist[nn_current])
        nxt = int(np.argmin(masked))
        nn_order.append(nxt)
        nn_visited[nxt] = True
        nn_current = nxt
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Model tour
    ax1.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, zorder=5)
    ax1.scatter(coords[0, 0], coords[0, 1], c='red', s=100, marker='*', zorder=6, label='Start')
    for i, (x, y) in enumerate(coords):
        ax1.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
    tour_coords = coords[order]
    ax1.plot(tour_coords[:, 0], tour_coords[:, 1], 'r-', alpha=0.7, linewidth=1.5)
    ax1.plot([tour_coords[-1, 0], tour_coords[0, 0]], [tour_coords[-1, 1], tour_coords[0, 1]], 'r-', alpha=0.7, linewidth=1.5)
    ax1.set_title(f"Model Tour (n={n})")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_aspect('equal')
    ax1.legend()
    
    # True NN tour
    ax2.scatter(coords[:, 0], coords[:, 1], c='blue', s=50, zorder=5)
    ax2.scatter(coords[0, 0], coords[0, 1], c='red', s=100, marker='*', zorder=6, label='Start')
    for i, (x, y) in enumerate(coords):
        ax2.annotate(str(i), (x, y), xytext=(3, 3), textcoords='offset points', fontsize=8)
    nn_coords = coords[nn_order]
    ax2.plot(nn_coords[:, 0], nn_coords[:, 1], 'g-', alpha=0.7, linewidth=1.5)
    ax2.plot([nn_coords[-1, 0], nn_coords[0, 0]], [nn_coords[-1, 1], nn_coords[0, 1]], 'g-', alpha=0.7, linewidth=1.5)
    ax2.set_title(f"True NN Tour (n={n})")
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_aspect('equal')
    ax2.legend()
    
    fig.tight_layout()
    return fig

def load_latest_results():
    """Load the most recent benchmark results for this attempt."""
    results_base = results_dir(__file__).parent
    if not results_base.exists():
        return None
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
    if not run_dirs:
        return None
    latest = run_dirs[-1]
    benchmark_path = latest / "benchmark.json"
    if benchmark_path.exists():
        import json
        with open(benchmark_path) as f:
            return json.load(f)
    return None

def create_metrics_plot(metrics):
    """Create a plot of metrics across problem sizes."""
    if metrics is None:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No benchmark results yet.\nRun main.py first.", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    sweep = metrics.get('sweep', [])
    if not sweep:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No sweep data in results.", ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    
    ns = [s['n'] for s in sweep]
    accs = [s['nn_accuracy'] for s in sweep]
    ratios = [s['tour_length_ratio'] for s in sweep]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(ns, accs, 'o-', label='Model', color='blue')
    ax1.set_xlabel('Number of Cities (N)')
    ax1.set_ylabel('Step-wise NN Accuracy')
    ax1.set_title('Nearest-Neighbor Accuracy vs Problem Size')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(ns, ratios, 'o-', label='Model', color='green')
    ax2.set_xlabel('Number of Cities (N)')
    ax2.set_ylabel('Tour Length Ratio (NN / Model)')
    ax2.set_title('Tour Quality vs Problem Size')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    fig.tight_layout()
    return fig

with gr.Blocks(title="attention_tsp: pass_2 - Attention-based NN Router") as demo:
    # Demo tab ---------------------------------------------------------------
    with gr.Tab(label="Demo: Attention NN Router"):
        gr.Markdown("""
        ## Hand-Coded Attention Mechanism for Nearest-Neighbor TSP
        
        This attempt implements the NN routing heuristic as an **attention mechanism**:
        
        - **Query**: Current city coordinates (shape 1×2)
        - **Keys**: All city coordinates (shape N×2)  
        - **Attention scores**: `-temperature * ||q - k||²` (negative squared Euclidean distance)
        - **Temperature**: 10.0 (sharpens attention to nearest)
        - **Visited masking**: Applied before argmax
        
        The mechanism runs entirely on GPU using PyTorch. The demo below visualizes
        the model's greedy tour vs the true nearest-neighbor tour for a selected instance.
        """)
        
        with gr.Row():
            n_selector = gr.Slider(minimum=5, maximum=40, value=10, step=5, label="Number of Cities (N)")
            seed_selector = gr.Number(value=123, label="Instance Seed", precision=0)
        
        run_btn = gr.Button("Run Tour Comparison", variant="primary")
        fig_out = gr.Plot(label="Tour Visualization")
        
        run_btn.click(
            fn=run_demo_tour,
            inputs=[n_selector, seed_selector],
            outputs=fig_out,
        )
        
        # Also show latest benchmark results
        gr.Markdown("### Latest Benchmark Results (this attempt)")
        refresh_btn = gr.Button("Refresh Results")
        metrics_plot = gr.Plot(label="Metrics Across Problem Sizes")
        
        def refresh_metrics():
            metrics = load_latest_results()
            return create_metrics_plot(metrics)
        
        refresh_btn.click(fn=refresh_metrics, outputs=metrics_plot)
        demo.load(fn=refresh_metrics, outputs=metrics_plot)
    
    # Benchmark tab ----------------------------------------------------------
    with gr.Tab(label="Benchmark: All Attempts"):
        gr.Markdown("""
        Leaderboard across all attempts for this goal. Shows `size_robustness` (headline metric),
        per-size NN accuracy, tour length ratios, and comparison to random baseline.
        """)
        # benchmark_panel creates its own UI inside the current Blocks context
        benchmark_panel("..")

if __name__ == "__main__":
    demo.launch(debug=True)

# Module-level export for boot-check
# (already exposed as `demo` via the `with gr.Blocks() as demo:` context)
