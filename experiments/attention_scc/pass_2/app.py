import gradio as gr
import json
import os
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel, results_dir

# Get the goal directory (parent of attempt directory)
current_dir = Path(__file__).parent
goal_dir = current_dir.parent

def find_latest_results(goal_dir: Path):
    """Find the most recent results directory across all attempts."""
    latest = None
    latest_time = ""
    
    for attempt_dir in goal_dir.iterdir():
        if not attempt_dir.is_dir() or attempt_dir.name.startswith("."):
            continue
        results_base = attempt_dir / "results"
        if not results_base.exists():
            continue
        for run_dir in results_base.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith("results_"):
                continue
            # Extract timestamp from directory name: results_YYYYMMDD_HHMMSS
            try:
                time_str = run_dir.name[8:]  # strip "results_"
                if time_str > latest_time:
                    latest_time = time_str
                    latest = run_dir
            except:
                pass
    return latest

def load_latest_payload():
    """Load the benchmark payload from the latest run."""
    run_dir = find_latest_results(goal_dir)
    if run_dir is None:
        return None
    bench_file = run_dir / "benchmark.json"
    if not bench_file.exists():
        return None
    with open(bench_file, "r") as f:
        return json.load(f)

def create_sweep_plot(payload):
    """Create a matplotlib figure of the SCC curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    sweep = payload["sweep"]
    rhos = [s["rho"] for s in sweep]
    means = [s["target_attention_mean"] for s in sweep]
    stds = [s["target_attention_std"] for s in sweep]
    chances = [s["chance_level"] for s in sweep]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot method performance with error bars
    ax.errorbar(rhos, means, yerr=stds, fmt='o-', capsize=5, 
                label='Standard Attention', color='tab:blue', linewidth=2, markersize=8)
    
    # Plot chance baseline
    ax.plot(rhos, chances, 's--', label='Chance (1/K)', color='tab:red', linewidth=1.5, markersize=6)
    
    ax.set_xlabel(r'Superposition Ratio $\rho = K/d$', fontsize=12)
    ax.set_ylabel('Target Attention Mass', fontsize=12)
    ax.set_title('Superposition Capacity Curve (SCC)', fontsize=14)
    ax.set_xscale('log', base=2)
    ax.set_xticks(rhos)
    ax.set_xticklabels([str(r) for r in rhos])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add AUC annotation
    from benchmark import score
    metrics = score(payload)
    auc_text = f'scc_auc = {metrics["scc_auc"]:.3f}\nlift = {metrics["lift_over_linear_auc"]:.3f}'
    ax.text(0.02, 0.98, auc_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def update_demo():
    """Load latest payload and generate plot."""
    payload = load_latest_payload()
    if payload is None:
        return None, "No benchmark results found. Run main.py first."
    fig = create_sweep_plot(payload)
    # Also return summary text
    sweep = payload["sweep"]
    summary_lines = ["## SCC Sweep Results\n"]
    for s in sweep:
        summary_lines.append(
            f"- ρ={s['rho']:.2f} (K={s['K']}): "
            f"target_attn={s['target_attention_mean']:.4f} ± {s['target_attention_std']:.4f}, "
            f"chance={s['chance_level']:.4f}"
        )
    summary = "\n".join(summary_lines)
    return fig, summary

with gr.Blocks() as demo:
    gr.Markdown("# Attention SCC: Superposition Capacity Curve")
    gr.Markdown(
        "Measures how well a single attention head can attend to a target key "
        "as the number of superimposed keys K grows relative to head dimension d=64."
    )
    
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                with gr.Column(scale=2):
                    plot = gr.Plot(label="SCC Curve")
                with gr.Column(scale=1):
                    summary = gr.Markdown()
            
            refresh_btn = gr.Button("Refresh from latest run", variant="secondary")
            refresh_btn.click(update_demo, inputs=[], outputs=[plot, summary])
            
            # Load on startup
            demo.load(update_demo, inputs=[], outputs=[plot, summary])
        
        with gr.TabItem("Benchmark"):
            gr.Markdown("## Benchmark Leaderboard")
            with gr.Blocks():
                benchmark_panel(str(goal_dir))

if __name__ == "__main__":
    demo.launch()