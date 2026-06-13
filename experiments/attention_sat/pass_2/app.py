import gradio as gr
import numpy as np
import json
from pathlib import Path
from agentic.experiments import benchmark_panel

# Goal directory for benchmark panel
goal_dir = Path(__file__).parent.parent

# Find latest run directory
results_dir = Path(__file__).parent / "results"
run_dirs = sorted(results_dir.glob("*")) if results_dir.exists() else []
latest_run = run_dirs[-1] if run_dirs else None


def load_latest_payload():
    """Load payload from latest run's benchmark.json or compute from sweep."""
    if latest_run is None:
        return None
    bench_file = latest_run / "benchmark.json"
    if bench_file.exists():
        with open(bench_file) as f:
            return json.load(f)
    return None


def create_saturation_plot(payload: dict):
    """Create a plot showing saturation metrics across logit scales."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    sweep = payload['sweep']
    scales = [r['logit_scale'] for r in sweep]
    sat_scores = [r['saturation_score'] for r in sweep]
    mean_entropies = [r['mean_entropy'] for r in sweep]
    max_weights = [r['max_attn_weight'] for r in sweep]
    ref_mean_entropies = [r['ref_mean_entropy'] for r in sweep]
    ref_max_weights = [r['ref_max_attn_weight'] for r in sweep]
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f'Attention Saturation Sweep - {latest_run.name if latest_run else "No run"}')
    
    # Saturation score vs scale
    ax = axes[0, 0]
    ax.plot(scales, sat_scores, 'o-', label='Attempt saturation_score', color='blue')
    ax.axvline(x=10.0, color='red', linestyle='--', label='Saturation threshold (10.0)')
    ax.set_xscale('log')
    ax.set_xlabel('Logit Scale')
    ax.set_ylabel('Saturation Score')
    ax.set_title('Saturation Score vs Logit Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean entropy vs scale (attempt vs reference)
    ax = axes[0, 1]
    ax.plot(scales, mean_entropies, 'o-', label='Attempt mean_entropy', color='blue')
    ax.plot(scales, ref_mean_entropies, 's--', label='Reference mean_entropy', color='orange')
    ax.axvline(x=10.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Logit Scale')
    ax.set_ylabel('Mean Entropy')
    ax.set_title('Mean Attention Entropy vs Logit Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Max attention weight vs scale
    ax = axes[1, 0]
    ax.plot(scales, max_weights, 'o-', label='Attempt max_weight', color='blue')
    ax.plot(scales, ref_max_weights, 's--', label='Reference max_weight', color='orange')
    ax.axvline(x=10.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Logit Scale')
    ax.set_ylabel('Mean Max Attention Weight')
    ax.set_title('Max Attention Weight vs Logit Scale')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # ROC-style: saturation score vs ground truth
    ax = axes[1, 1]
    saturated = [1 if s >= 10.0 else 0 for s in scales]
    colors = ['green' if s == 0 else 'red' for s in saturated]
    ax.scatter(scales, sat_scores, c=colors, s=100, alpha=0.7, label='Attempt score')
    ax.axvline(x=10.0, color='red', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Logit Scale')
    ax.set_ylabel('Saturation Score')
    ax.set_title('Saturation Detection (red=saturated, green=non-sat)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_attention_heatmap(payload: dict, scale_idx: int = 4):
    """Create attention weight heatmap for a specific scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    sweep = payload['sweep']
    record = sweep[scale_idx]
    scale = record['logit_scale']
    
    # Average over batch dimension
    attn_weights = record['attn_weights']  # (batch, seq_len, seq_len)
    mean_attn = attn_weights.mean(axis=0)  # (seq_len, seq_len)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_attn, cmap='hot', aspect='auto')
    ax.set_title(f'Mean Attention Weights (logit_scale={scale})')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# Attention Saturation Detection - pass_2")
    gr.Markdown(
        "This attempt computes **exact softmax attention on GPU** and uses "
        "**mean max attention weight** as the saturation score. "
        "The saturation threshold is `logit_scale >= 10.0`."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            run_dropdown = gr.Dropdown(
                choices=[d.name for d in run_dirs],
                value=latest_run.name if latest_run else None,
                label="Select Run"
            )
            load_btn = gr.Button("Load Run", variant="primary")
        
        with gr.Column(scale=1):
            scale_slider = gr.Slider(
                minimum=0, maximum=6, step=1, value=4,
                label="Logit Scale Index (for heatmap)",
                info="0=0.1, 1=0.3, 2=1.0, 3=3.0, 4=10.0, 5=30.0, 6=100.0"
            )
    
    with gr.Row():
        saturation_plot = gr.Plot(label="Saturation Metrics Across Scales")
    
    with gr.Row():
        heatmap_plot = gr.Plot(label="Attention Weight Heatmap")
    
    with gr.Row():
        metrics_json = gr.JSON(label="Benchmark Metrics")
    
    def load_run(run_name: str):
        if not run_name:
            return None, None, None, {}
        run_path = results_dir / run_name
        bench_file = run_path / "benchmark.json"
        if not bench_file.exists():
            return None, None, None, {}
        
        with open(bench_file) as f:
            bench = json.load(f)
        
        # Also need the payload for detailed sweep data
        # The payload is not saved separately, but we can reconstruct key parts from benchmark
        # For now, load from the sweep saved in the run directory if available
        payload_file = run_path / "payload.json"
        if payload_file.exists():
            with open(payload_file) as f:
                payload = json.load(f)
        else:
            # Reconstruct minimal payload from benchmark
            payload = {'sweep': []}
        
        fig1 = create_saturation_plot(payload) if payload.get('sweep') else None
        fig2 = create_attention_heatmap(payload, 4) if payload.get('sweep') else None
        
        return fig1, fig2, bench
    
    def update_heatmap(payload, scale_idx):
        if not payload or not payload.get('sweep'):
            return None
        return create_attention_heatmap(payload, scale_idx)
    
    # Load latest run on startup
    demo.load(
        load_run,
        inputs=[run_dropdown],
        outputs=[saturation_plot, heatmap_plot, metrics_json]
    )
    
    load_btn.click(
        load_run,
        inputs=[run_dropdown],
        outputs=[saturation_plot, heatmap_plot, metrics_json]
    )
    
    scale_slider.change(
        update_heatmap,
        inputs=[gr.State(value=load_latest_payload()), scale_slider],
        outputs=[heatmap_plot]
    )

    # Benchmark tab
    with gr.Tab("Benchmark"):
        benchmark_panel(goal_dir)


if __name__ == "__main__":
    demo.launch()