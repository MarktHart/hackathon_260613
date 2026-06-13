import json
import gradio as gr
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
ATTEMPT_DIR = Path(__file__).parent
RESULTS_DIR = ATTEMPT_DIR / "results"


def get_available_runs():
    """Return list of (run_name, run_path) sorted by name (timestamp) descending."""
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for run_dir in RESULTS_DIR.iterdir():
        if run_dir.is_dir():
            benchmark_file = run_dir / "benchmark.json"
            if benchmark_file.exists():
                runs.append((run_dir.name, run_dir))
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs


def load_payload(run_path: Path) -> dict:
    """Load the benchmark.json payload for a run."""
    with open(run_path / "benchmark.json", "r") as f:
        return json.load(f)


def create_plot(payload: dict):
    """Create matplotlib figure showing attention distance vs edit distance."""
    sweep = payload.get("sweep", [])
    if not sweep:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No sweep data available", ha="center", va="center", transform=ax.transAxes)
        return fig
    
    edit_dists = [s["edit_distance"] for s in sweep]
    attn_means = [s["attn_distance_mean"] for s in sweep]
    attn_stds = [s["attn_distance_std"] for s in sweep]
    n_pairs = [s["n_pairs"] for s in sweep]
    
    # Baseline
    baseline_means = payload.get("linear_baseline", {}).get("attn_distance_mean", [])
    baseline_stds = payload.get("linear_baseline", {}).get("attn_distance_std", [])
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Plot model attention distance with error bars
    ax.errorbar(edit_dists, attn_means, yerr=attn_stds, 
                fmt='o-', label="GPT-2 Layer 5 Head 3", 
                color="#1f77b4", capsize=4, linewidth=2, markersize=8)
    
    # Plot baseline
    if baseline_means:
        ax.errorbar(edit_dists, baseline_means, yerr=baseline_stds,
                    fmt='s--', label="Random Attention Baseline",
                    color="#ff7f0e", capsize=4, linewidth=1.5, markersize=6, alpha=0.7)
    
    ax.set_xlabel("Levenshtein Edit Distance", fontsize=12)
    ax.set_ylabel("Attention Distance (1 - Cosine Similarity)", fontsize=12)
    ax.set_title(f"Attention Distance vs Edit Distance\n"
                 f"{payload.get('model_name', 'gpt2')} Layer {payload.get('layer', 5)} Head {payload.get('head', 3)}", 
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, max(edit_dists) + 1))
    
    # Add n_pairs as text annotations
    for i, (d, n) in enumerate(zip(edit_dists, n_pairs)):
        ax.annotate(f"n={n}", (edit_dists[i], attn_means[i]), 
                    textcoords="offset points", xytext=(0, 10), 
                    ha='center', fontsize=8, color="#1f77b4")
    
    plt.tight_layout()
    return fig


def update_demo(run_name: str):
    """Update the demo plot and summary text for selected run."""
    runs = get_available_runs()
    run_dict = {name: path for name, path in runs}
    
    if run_name not in run_dict:
        return None, "Run not found"
    
    payload = load_payload(run_dict[run_name])
    fig = create_plot(payload)
    
    # Summary text
    sweep = payload.get("sweep", [])
    if sweep:
        edit_dists = [s["edit_distance"] for s in sweep]
        attn_means = [s["attn_distance_mean"] for s in sweep]
        
        # Compute Spearman correlation
        from scipy.stats import spearmanr
        try:
            rho, p = spearmanr(edit_dists, attn_means)
            corr_text = f"Spearman ρ = {rho:.4f} (p = {p:.4g})"
        except:
            corr_text = "Spearman ρ = N/A"
        
        summary = (f"**Run:** {run_name}\n"
                   f"**Model:** {payload.get('model_name', 'gpt2')} "
                   f"(Layer {payload.get('layer', 5)}, Head {payload.get('head', 3)})\n"
                   f"**Sequence Length:** {payload.get('seq_len', 32)}\n"
                   f"**Vocabulary Size:** {payload.get('vocab_size', 100)}\n"
                   f"**Correlation:** {corr_text}\n"
                   f"**Data Points:** {len(sweep)} edit distance buckets")
    else:
        summary = "No sweep data available"
    
    return fig, summary


with gr.Blocks(title="Attention Edit Distance - First Pass") as demo:
    gr.Markdown("# Attention Edit Distance: First Pass\n"
                "Does attention pattern distance correlate with Levenshtein edit distance? "
                "Using GPT-2 small, Layer 5, Head 3.")
    
    with gr.Row():
        with gr.Column(scale=1):
            runs = get_available_runs()
            run_choices = [name for name, _ in runs]
            run_dropdown = gr.Dropdown(
                choices=run_choices,
                value=run_choices[0] if run_choices else None,
                label="Select Run",
                interactive=True
            )
            gr.Markdown("### Run Summary")
            summary_md = gr.Markdown("Select a run to see summary")
        
        with gr.Column(scale=2):
            plot_output = gr.Plot(label="Attention Distance vs Edit Distance")
    
    def on_run_change(run_name):
        if run_name is None:
            return None, "No runs available"
        return update_demo(run_name)
    
    run_dropdown.change(
        fn=on_run_change,
        inputs=[run_dropdown],
        outputs=[plot_output, summary_md]
    )
    
    # Initial load
    if run_choices:
        demo.load(
            fn=lambda: update_demo(run_choices[0]),
            inputs=[],
            outputs=[plot_output, summary_md]
        )
    
    gr.Markdown("---")
    
    # Benchmark panel - shows leaderboard across all attempts
    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()