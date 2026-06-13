import gradio as gr
import numpy as np
import json
from pathlib import Path
from agentic.experiments import benchmark_panel

# Locate the goal directory (parent of this attempt)
GOAL_DIR = Path(__file__).parent.parent

def load_latest_run():
    """Find the most recent run directory and load its artefacts."""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None, None, None
    
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        return None, None, None
    
    latest = run_dirs[-1]
    
    # Load benchmark.json
    bench_path = latest / "benchmark.json"
    if bench_path.exists():
        with open(bench_path) as f:
            benchmark = json.load(f)
    else:
        benchmark = None
    
    # Load payload from task evaluation (if saved)
    # For now, we'll recompute from the sweep data in benchmark
    payload = benchmark.get("payload") if benchmark else None
    
    return latest, benchmark, payload


def make_perm_heatmap(attn_weights, head_idx=0):
    """Create a heatmap figure for attention weights of one head."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # attn_weights: (batch, heads, seq, seq)
    # Average over batch for display
    avg_attn = attn_weights[:, head_idx].mean(axis=0)
    
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_attn, cmap='viridis', aspect='auto')
    ax.set_title(f"Head {head_idx} Average Attention Weights")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    return fig


def make_equivariance_chart(benchmark):
    """Create bar chart of equivariance errors per permutation."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if not benchmark or "payload" not in benchmark:
        return None
    
    payload = benchmark["payload"]
    sweep = payload.get("sweep", [])
    
    perm_ids = [s["perm_id"] for s in sweep]
    attn_errors = [s["attn_fro_error"] for s in sweep]
    out_errors = [s["output_fro_error"] for s in sweep]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Attention equivariance
    bars1 = ax1.bar([str(p) for p in perm_ids], attn_errors, color=['green' if p == 0 else 'blue' for p in perm_ids])
    ax1.set_title("Attention Equivariance Error (Frobenius)")
    ax1.set_xlabel("Permutation ID (0=identity)")
    ax1.set_ylabel("Relative Frobenius Error")
    ax1.set_yscale('log')
    
    # Output equivariance
    bars2 = ax2.bar([str(p) for p in perm_ids], out_errors, color=['green' if p == 0 else 'orange' for p in perm_ids])
    ax2.set_title("Output Equivariance Error (Frobenius)")
    ax2.set_xlabel("Permutation ID (0=identity)")
    ax2.set_ylabel("Relative Frobenius Error")
    ax2.set_yscale('log')
    
    plt.tight_layout()
    return fig


def make_head_comparison_chart(benchmark):
    """Create chart comparing equivariance across heads."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # We don't have per-head data in the payload, so show aggregate
    if not benchmark:
        return None
    
    metrics = benchmark.get("metrics", {})
    
    # Extract canonical metrics
    keys = [k for k in metrics.keys() if k.startswith("equivariance_fro_perm_") or k.startswith("output_equivariance_fro_perm_")]
    
    if not keys:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 4))
    
    attn_keys = sorted([k for k in keys if k.startswith("equivariance_fro_perm_")])
    out_keys = sorted([k for k in keys if k.startswith("output_equivariance_fro_perm_")])
    
    x = range(len(attn_keys))
    attn_vals = [metrics[k] for k in attn_keys]
    out_vals = [metrics[k] for k in out_keys]
    
    width = 0.35
    ax.bar([i - width/2 for i in x], attn_vals, width, label='Attention', alpha=0.8, color='blue')
    ax.bar([i + width/2 for i in x], out_vals, width, label='Output', alpha=0.8, color='orange')
    
    ax.set_xticks(x)
    ax.set_xticklabels([k.split('_')[-1] for k in attn_keys])
    ax.set_xlabel("Permutation ID")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_title("Equivariance Error by Permutation")
    ax.set_yscale('log')
    ax.legend()
    plt.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# Attention Equality: Permutation Equivariance Demo")
    
    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown("""
            This demo visualises the permutation equivariance of a multi-head self-attention mechanism.
            The model is a standard transformer-style attention block with fixed random weights (no training).
            
            **Permutation Equivariance Property:**
            - For a permutation matrix P, attention weights should satisfy: A(PX) = P A(X) Pᵀ
            - Output embeddings should satisfy: Out(PX) = P Out(X)
            
            Select a run to visualise its equivariance errors and attention patterns.
            """)
            
            with gr.Row():
                run_dropdown = gr.Dropdown(
                    label="Select Run",
                    choices=[],
                    interactive=True,
                )
                refresh_btn = gr.Button("Refresh Runs")
            
            with gr.Row():
                equiv_plot = gr.Plot(label="Equivariance Errors by Permutation")
            
            with gr.Row():
                head_plot = gr.Plot(label="Per-Head / Per-Perm Comparison")
            
            with gr.Row():
                attn_heatmap = gr.Plot(label="Average Attention Weights (Head 0)")
            
            def update_run_list():
                results_dir = Path(__file__).parent / "results"
                if not results_dir.exists():
                    return gr.Dropdown(choices=[], value=None)
                runs = sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)
                return gr.Dropdown(choices=runs, value=runs[0] if runs else None)
            
            def load_run(run_name):
                if not run_name:
                    return None, None, None
                run_dir = Path(__file__).parent / "results" / run_name
                
                # Load benchmark
                bench_path = run_dir / "benchmark.json"
                if not bench_path.exists():
                    return None, None, None
                
                with open(bench_path) as f:
                    benchmark = json.load(f)
                
                # Re-run model to get attention weights for visualization
                # We need to regenerate the batch and run model_fn
                # Import task and model
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from main import make_attention_model
                from task import generate
                
                batch = generate(seed=0)
                model_fn = make_attention_model(seed=42)
                
                # Get attention weights on original tokens
                orig_out = model_fn(batch.tokens)
                attn_weights = orig_out["attn_weights"]
                
                # Generate charts
                equiv_fig = make_equivariance_chart(benchmark)
                head_fig = make_head_comparison_chart(benchmark)
                heatmap_fig = make_perm_heatmap(attn_weights, head_idx=0)
                
                return equiv_fig, head_fig, heatmap_fig
            
            refresh_btn.click(update_run_list, outputs=run_dropdown)
            run_dropdown.change(load_run, inputs=run_dropdown, outputs=[equiv_plot, head_plot, attn_heatmap])
            
            # Initial load
            demo.load(update_run_list, outputs=run_dropdown)
        
        with gr.TabItem("Benchmark"):
            gr.Markdown("""
            ## Benchmark History
            This panel shows the benchmark metrics across all attempts and runs for the `attention_equality` goal.
            """)
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()