import gradio as gr
import numpy as np
import torch
from pathlib import Path

from agentic.experiments import load_task, benchmark_panel

DEVICE = "cuda"

def model_fn(pattern: np.ndarray, embed: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Hand-built attention mechanism for regex-like pattern matching (same as main.py)."""
    pattern_t = torch.as_tensor(pattern, dtype=torch.long, device=DEVICE)
    embed_t = torch.as_tensor(embed, dtype=torch.float32, device=DEVICE)
    residual_t = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    
    N, d = residual_t.shape
    L = pattern_t.shape[0]
    
    logits = torch.zeros(N, dtype=torch.float32, device=DEVICE)
    
    mask = (pattern_t != -1)
    if not mask.any():
        return logits.detach().cpu().numpy()
    
    concrete_idx = torch.where(mask)[0]
    concrete_tokens = pattern_t[concrete_idx]
    concrete_embeds = embed_t[concrete_tokens]
    
    for k, j in enumerate(concrete_idx):
        shift = L - 1 - j
        start_res = j
        end_res = N - L + j + 1
        if start_res < end_res:
            residual_slice = residual_t[start_res:end_res]
            target_embed = concrete_embeds[k]
            sims = residual_slice @ target_embed
            logits_start = start_res + shift
            logits_end = logits_start + sims.shape[0]
            logits[logits_start:logits_end] += sims
    
    num_concrete = concrete_idx.shape[0]
    logits = logits / num_concrete
    
    return logits.detach().cpu().numpy()


def load_latest_run():
    """Find the most recent run directory and load its data."""
    results_base = Path(__file__).parent / "results"
    if not results_base.exists():
        return None, None, None, None
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
    if not run_dirs:
        return None, None, None, None
    latest = run_dirs[-1]
    # We'll regenerate the batch and run model_fn for visualization
    task = load_task(__file__)
    batch = task.generate(seed=task.EVAL_SEED)
    return batch, latest


def compute_viz_data(batch, length_idx, seed_idx):
    """Compute attention scores and labels for a specific example."""
    if batch is None:
        return None
    
    # Flatten index: length_idx * N_SEEDS + seed_idx
    flat_idx = length_idx * task.N_SEEDS + seed_idx
    if flat_idx >= len(batch.patterns):
        return None
    
    pattern = batch.patterns[flat_idx]
    embed = batch.embeds[flat_idx]
    residual = batch.residuals[flat_idx]
    labels = batch.labels[flat_idx]
    L = batch.lengths[flat_idx]
    
    logits = model_fn(pattern, embed, residual)
    attn = np.exp(logits - logits.max())
    attn = attn / attn.sum()
    
    return {
        "pattern": pattern,
        "attn": attn,
        "labels": labels,
        "L": L,
        "logits": logits,
    }


def make_plot(viz_data):
    """Create a matplotlib figure showing attention vs ground truth."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    if viz_data is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center', transform=ax.transAxes)
        return fig
    
    attn = viz_data["attn"]
    labels = viz_data["labels"]
    pattern = viz_data["pattern"]
    L = viz_data["L"]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Attention scores
    N = len(attn)
    x = np.arange(N)
    ax1.plot(x, attn, 'b-', alpha=0.7, label='Attention')
    ax1.fill_between(x, 0, attn, alpha=0.3, color='blue')
    
    # Mark match-end positions
    match_positions = np.where(labels)[0]
    if len(match_positions) > 0:
        ax1.scatter(match_positions, attn[match_positions], color='red', s=50, zorder=5, label='True match ends')
    
    # Uniform baseline
    ax1.axhline(y=1.0/N, color='gray', linestyle='--', alpha=0.5, label='Uniform')
    
    ax1.set_ylabel('Attention weight')
    ax1.set_title(f'Pattern length L={L}, pattern={pattern.tolist()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Ground truth labels
    ax2.bar(x, labels.astype(float), width=1.0, color='red', alpha=0.5, label='Match end')
    ax2.set_ylabel('Label')
    ax2.set_xlabel('Position')
    ax2.set_ylim(0, 1.2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Load task once for constants
task = load_task(__file__)

# Load initial batch
initial_batch, initial_run_dir = load_latest_run()

with gr.Blocks() as demo:
    gr.Markdown("# Attention Regex: First Pass Demo")
    gr.Markdown("Visualizing hand-built attention mechanism for wildcard-capable pattern matching.")
    
    with gr.Row():
        with gr.Column(scale=1):
            run_dropdown = gr.Dropdown(
                choices=[d.name for d in sorted((Path(__file__).parent / "results").iterdir()) if d.is_dir()] if (Path(__file__).parent / "results").exists() else [],
                value=initial_run_dir.name if initial_run_dir else None,
                label="Run directory"
            )
            length_dropdown = gr.Dropdown(
                choices=[f"L={L}" for L in task.LENGTH_SWEEP],
                value=f"L={task.CANONICAL_LENGTH}",
                label="Pattern length"
            )
            seed_slider = gr.Slider(
                minimum=0, maximum=task.N_SEEDS - 1, step=1, value=0,
                label="Seed index"
            )
            refresh_btn = gr.Button("Refresh runs")
        
        with gr.Column(scale=3):
            plot_output = gr.Plot()
    
    def update_plot(run_name, length_str, seed_idx):
        batch, _ = load_latest_run()
        if batch is None:
            return None
        L = int(length_str.split("=")[1])
        length_idx = task.LENGTH_SWEEP.index(L)
        viz_data = compute_viz_data(batch, length_idx, int(seed_idx))
        return make_plot(viz_data)
    
    def refresh_runs():
        results_base = Path(__file__).parent / "results"
        if not results_base.exists():
            return gr.update(choices=[], value=None)
        run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
        choices = [d.name for d in run_dirs]
        value = choices[-1] if choices else None
        return gr.update(choices=choices, value=value)
    
    # Event handlers INSIDE the Blocks context
    demo.load(
        fn=lambda: make_plot(compute_viz_data(initial_batch, task.LENGTH_SWEEP.index(task.CANONICAL_LENGTH), 0)) if initial_batch else None,
        outputs=plot_output
    )
    length_dropdown.change(
        fn=update_plot,
        inputs=[run_dropdown, length_dropdown, seed_slider],
        outputs=plot_output
    )
    seed_slider.change(
        fn=update_plot,
        inputs=[run_dropdown, length_dropdown, seed_slider],
        outputs=plot_output
    )
    run_dropdown.change(
        fn=update_plot,
        inputs=[run_dropdown, length_dropdown, seed_slider],
        outputs=plot_output
    )
    refresh_btn.click(
        fn=refresh_runs,
        outputs=run_dropdown
    ).then(
        fn=update_plot,
        inputs=[run_dropdown, length_dropdown, seed_slider],
        outputs=plot_output
    )
    
    # Benchmark tab
    with gr.Tab("Benchmark"):
        gr.Markdown("## Benchmark History Across Attempts")
        benchmark_panel(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    demo.launch()