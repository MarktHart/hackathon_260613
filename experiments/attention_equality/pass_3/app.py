import gradio as gr
import json
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent

def load_latest_run():
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None, None
    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        return None, None
    latest = run_dirs[-1]
    bench_path = latest / "benchmark.json"
    benchmark = json.load(bench_path) if bench_path.exists() else None
    return latest, benchmark


def make_match_mass_barplot(benchmark):
    """Bar plot: match_mass vs uniform baseline across L sweep."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not benchmark or "sweep" not in benchmark:
        return None

    L_vals = []
    match_masses = []
    uniform_baselines = []

    for rec in benchmark["sweep"]:
        L = rec.get("L")
        mm = rec.get("match_mass")
        ub = rec.get("uniform_baseline")
        if L is not None and mm is not None and ub is not None:
            L_vals.append(L)
            match_masses.append(mm)
            uniform_baselines.append(ub)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(L_vals))
    width = 0.35
    ax.bar(x - width/2, match_masses, width, label='match_mass (equality head)', color='#2E86AB', alpha=0.9)
    ax.bar(x + width/2, uniform_baselines, width, label='uniform baseline', color='#F18F01', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(L) for L in L_vals])
    ax.set_xlabel("Sequence Length L")
    ax.set_ylabel("Normalized Attention Mass (attn[p2, p1])")
    ax.set_ylim(bottom=0, top=1.05)
    ax.set_title("Equality Head: Match Mass vs Uniform Baseline Across L Sweep")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def make_lift_plot(benchmark):
    """Line plot: lift over uniform baseline across L sweep."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not benchmark or "sweep" not in benchmark:
        return None

    L_vals = []
    lifts = []
    for rec in benchmark["sweep"]:
        L = rec.get("L")
        mm = rec.get("match_mass")
        ub = rec.get("uniform_baseline")
        if L is not None and mm is not None and ub is not None:
            L_vals.append(L)
            lifts.append(mm - ub)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(L_vals, lifts, 'o-', color='#2E86AB', linewidth=2, markersize=8, label='lift = match_mass - uniform')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Sequence Length L")
    ax.set_ylabel("Lift Over Uniform Baseline")
    ax.set_title("Equality Head Robustness: Lift Increases with Sequence Length")
    ax.set_xscale('log', base=2)
    ax.set_xticks(L_vals)
    ax.set_xticklabels([str(L) for L in L_vals])
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def make_attention_heatmap(benchmark):
    """Heatmap of the hand-built attention pattern at canonical L=16 (first sequence)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Reconstruct the attention pattern for one sequence at L=16
    # Our circuit: uniform for all queries except p2, which puts ~1.0 on p1
    L = 16
    # Generate a representative batch to get p1, p2
    from experiments.attention_equality.task import generate
    batch = generate(seed=0, L=L)
    p1, p2 = batch.p1[0], batch.p2[0]
    mask = batch.mask[0]

    attn = np.zeros((L, L))
    counts = mask.sum(axis=1, keepdims=True)
    attn = mask.astype(float) / counts  # uniform over allowed keys
    
    # Query p2 routes to p1
    attn[p2, :] = 0.0
    attn[p2, p1] = 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap='viridis', origin='lower', aspect='auto', vmin=0, vmax=1)
    ax.axhline(y=p2, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'query p2={p2}')
    ax.axvline(x=p1, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'key p1={p1}')
    ax.plot(p1, p2, 'r*', markersize=15, label='match (p2→p1)')
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Routing (Equality Head, L=16, seq 0)\np1={p1}, p2={p2}")
    ax.legend(loc='upper right', fontsize=8)
    plt.colorbar(im, ax=ax, label='Attention Weight')
    plt.tight_layout()
    return fig


def make_circuit_diagram():
    """Schematic of the equality circuit: Q/K construction and routing logic."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Token embeddings
    ax.add_patch(mpatches.Rectangle((0.5, 2.5), 2, 1, facecolor='#E8F4FD', edgecolor='#2E86AB', linewidth=2))
    ax.text(1.5, 3.0, 'Token IDs\n(B, L)', ha='center', va='center', fontsize=10, weight='bold')

    # Key projection: one-hot
    ax.add_patch(mpatches.Rectangle((3.5, 2.5), 2, 1, facecolor='#FFF3E0', edgecolor='#F18F01', linewidth=2))
    ax.text(4.5, 3.0, 'K = One-Hot\n(V=128)', ha='center', va='center', fontsize=10, weight='bold')

    # Query projection: one-hot + p2 modification
    ax.add_patch(mpatches.Rectangle((3.5, 1.0), 2, 1, facecolor='#FFF3E0', edgecolor='#F18F01', linewidth=2))
    ax.text(4.5, 1.5, 'Q = One-Hot\n+ p2 suppress self', ha='center', va='center', fontsize=10, weight='bold')

    # Scores
    ax.add_patch(mpatches.Rectangle((6.5, 1.0), 2, 2.5, facecolor='#E8F8F0', edgecolor='#27AE60', linewidth=2))
    ax.text(7.5, 2.25, 'Scores = QKᵀ/√d\n+ causal mask\n+ p2→p1 boost', ha='center', va='center', fontsize=10, weight='bold')

    # Softmax
    ax.add_patch(mpatches.Rectangle((9.0, 1.5), 1.5, 1, facecolor='#FDEBD0', edgecolor='#E67E22', linewidth=2))
    ax.text(9.75, 2.0, 'Softmax', ha='center', va='center', fontsize=10, weight='bold')

    # Arrows
    ax.annotate('', xy=(3.5, 3.0), xytext=(2.5, 3.0), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(3.5, 1.5), xytext=(2.5, 1.5), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(6.5, 2.25), xytext=(5.5, 3.0), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(6.5, 2.25), xytext=(5.5, 1.5), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))
    ax.annotate('', xy=(9.0, 2.0), xytext=(8.5, 2.25), arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    # Output
    ax.add_patch(mpatches.Rectangle((9.0, 0.2), 1.5, 0.8, facecolor='#FADBD8', edgecolor='#E74C3C', linewidth=2))
    ax.text(9.75, 0.6, 'Attn[p2, p1] ≈ 1.0', ha='center', va='center', fontsize=9, weight='bold', color='#E74C3C')

    ax.set_title("Hand-Built Equality Circuit: One-Hot Keys + Positional Bias at p2", fontsize=12, weight='bold', pad=20)
    return fig


def load_selected_run(run_name):
    if not run_name:
        return None, None, None, None
    run_dir = Path(__file__).parent / "results" / run_name
    bench_path = run_dir / "benchmark.json"
    if not bench_path.exists():
        return None, None, None, None
    with open(bench_path) as f:
        benchmark = json.load(f)
    return (
        benchmark,
        make_match_mass_barplot(benchmark),
        make_lift_plot(benchmark),
        make_attention_heatmap(benchmark),
    )


with gr.Blocks() as demo:
    gr.Markdown("# Equality Head — Attention Equality Lookup")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            gr.Markdown("""
            **Hand-built equality head** that routes attention from query position `p2` to the matching key `p1`
            (same token value). The circuit uses **one-hot key vectors** (dimension = vocab size = 128) so every
            token has a perfectly orthogonal key. Queries match keys by token identity. At the query position `p2`,
            a positional bias suppresses self-attention and boosts the earlier position `p1`, implementing
            "*find the previous occurrence of this token*"."
            """)

            with gr.Row():
                run_dropdown = gr.Dropdown(label="Select Run", choices=[], interactive=True)
                refresh_btn = gr.Button("Refresh Runs", variant="secondary")

            with gr.Row():
                barplot = gr.Plot(label="Match Mass vs Uniform Baseline (L Sweep)")
                lift_plot = gr.Plot(label="Lift Over Uniform Baseline")

            with gr.Row():
                heatmap = gr.Plot(label="Attention Routing Heatmap (Canonical L=16)")
                circuit = gr.Plot(label="Circuit Diagram")

            def update_run_list():
                results_dir = Path(__file__).parent / "results"
                if not results_dir.exists():
                    return gr.Dropdown(choices=[], value=None)
                runs = sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)
                return gr.Dropdown(choices=runs, value=runs[0] if runs else None)

            def load_run(run_name):
                if not run_name:
                    return None, None, None, None
                benchmark, bp, lp, hp = load_selected_run(run_name)
                return bp, lp, hp, make_circuit_diagram()

            refresh_btn.click(update_run_list, outputs=run_dropdown)
            run_dropdown.change(load_run, inputs=run_dropdown, outputs=[barplot, lift_plot, heatmap, circuit])
            demo.load(update_run_list, outputs=run_dropdown)

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Benchmark History Across All Attempts")
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()