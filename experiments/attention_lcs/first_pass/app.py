import gradio as gr
import numpy as np
import json
from pathlib import Path

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent


def load_latest_run():
    """Find and load the most recent run directory."""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return None, None

    run_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    if not run_dirs:
        return None, None

    latest = run_dirs[-1]
    benchmark_file = latest / "benchmark.json"
    if benchmark_file.exists():
        with benchmark_file.open() as f:
            benchmark = json.load(f)
    else:
        benchmark = None

    return latest, benchmark


def load_run(run_name):
    """Load a specific run by name."""
    run_dir = Path(__file__).parent / "results" / run_name
    benchmark_file = run_dir / "benchmark.json"
    if benchmark_file.exists():
        with benchmark_file.open() as f:
            benchmark = json.load(f)
    else:
        benchmark = None
    return benchmark


def get_run_names():
    """Get list of available run directories."""
    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        return []
    return sorted([d.name for d in results_dir.iterdir() if d.is_dir()], reverse=True)


def plot_head_masses(benchmark):
    """Create a bar chart of LCS attention mass per head."""
    if not benchmark or "sweep" not in benchmark:
        return None

    heads = [rec["head"] for rec in benchmark["sweep"]]
    masses = [rec["lcs_attention_mass"] for rec in benchmark["sweep"]]
    lifts = [rec["lcs_lift"] for rec in benchmark["sweep"]]
    baseline = benchmark.get("random_baseline_mass", 0)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Attention mass per head
    bars1 = ax1.bar(heads, masses, color="steelblue", alpha=0.7, label="LCS attention mass")
    ax1.axhline(y=baseline, color="red", linestyle="--", label=f"Random baseline ({baseline:.3f})")
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Attention mass on LCS keys")
    ax1.set_title("LCS Attention Mass per Head")
    ax1.set_ylim(0, max(1.0, max(masses) * 1.2) if masses else 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lift per head
    bars2 = ax2.bar(heads, lifts, color="orange", alpha=0.7, label="Lift over baseline")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.set_xlabel("Head")
    ax2.set_ylabel("Lift (mass - baseline)")
    ax2.set_title("LCS Lift per Head")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_attention_heatmap(benchmark, head_idx=0):
    """Create a heatmap of attention for a specific head (placeholder - would need raw attention)."""
    # Note: We don't save raw attention weights, only the aggregated metrics.
    # This is a placeholder showing we'd visualize the attention pattern if we had it.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5, "Raw attention weights not saved.\nRun main.py with --save-raw to enable.",
            ha="center", va="center", transform=ax.transAxes, fontsize=12)
    ax.set_title(f"Attention Heatmap - Head {head_idx}")
    ax.axis("off")
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# Attention LCS - First Pass Attempt")

    with gr.Tab("Demo"):
        with gr.Row():
            run_dropdown = gr.Dropdown(
                choices=get_run_names(),
                label="Select Run",
                value=get_run_names()[0] if get_run_names() else None
            )
            refresh_btn = gr.Button("Refresh Runs", size="sm")

        with gr.Row():
            summary_md = gr.Markdown()

        with gr.Row():
            mass_plot = gr.Plot(label="LCS Attention Mass per Head")
            lift_plot = gr.Plot(label="LCS Lift per Head")

        with gr.Row():
            heatmap_head = gr.Slider(0, 3, value=0, step=1, label="Head for Heatmap")
            heatmap_plot = gr.Plot(label="Attention Heatmap")

        def update_demo(run_name):
            benchmark = load_run(run_name)
            if not benchmark:
                return "No benchmark data found.", None, None, None

            # Summary markdown
            config = benchmark.get("config", {})
            sweep = benchmark.get("sweep", [])
            baseline = benchmark.get("random_baseline_mass", 0)
            best_lift = max([r["lcs_lift"] for r in sweep]) if sweep else 0
            best_mass = max([r["lcs_attention_mass"] for r in sweep]) if sweep else 0

            summary = f"""
            **Run:** {run_name}
            **Config:** seq_len={config.get('seq_len')}, vocab_size={config.get('vocab_size')},
            num_examples={config.get('num_examples')}, seed={config.get('seed')},
            n_heads={config.get('n_heads')}

            **Random baseline mass:** {baseline:.4f}
            **Best head LCS mass:** {best_mass:.4f}
            **Best head lift:** {best_lift:.4f}
            **Robustness:** {best_lift / (1 - baseline) if baseline < 1 else 0:.4f}
            """

            fig_mass, fig_lift = create_plots(benchmark)
            fig_heatmap = plot_attention_heatmap(benchmark, 0)

            return summary, fig_mass, fig_lift, fig_heatmap

        def create_plots(benchmark):
            if not benchmark or "sweep" not in benchmark:
                return None, None

            heads = [rec["head"] for rec in benchmark["sweep"]]
            masses = [rec["lcs_attention_mass"] for rec in benchmark["sweep"]]
            lifts = [rec["lcs_lift"] for rec in benchmark["sweep"]]
            baseline = benchmark.get("random_baseline_mass", 0)

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # Mass plot
            fig1, ax1 = plt.subplots(figsize=(5, 4))
            ax1.bar(heads, masses, color="steelblue", alpha=0.7)
            ax1.axhline(y=baseline, color="red", linestyle="--", label=f"Random baseline ({baseline:.3f})")
            ax1.set_xlabel("Head")
            ax1.set_ylabel("Attention mass on LCS keys")
            ax1.set_title("LCS Attention Mass per Head")
            ax1.set_ylim(0, max(1.0, max(masses) * 1.2) if masses else 1.0)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Lift plot
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.bar(heads, lifts, color="orange", alpha=0.7)
            ax2.axhline(y=0, color="black", linewidth=0.5)
            ax2.set_xlabel("Head")
            ax2.set_ylabel("Lift (mass - baseline)")
            ax2.set_title("LCS Lift per Head")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig1, fig2

        def update_heatmap(run_name, head_idx):
            benchmark = load_run(run_name)
            return plot_attention_heatmap(benchmark, int(head_idx))

        # Event handlers inside the Blocks context
        run_dropdown.change(
            update_demo,
            inputs=[run_dropdown],
            outputs=[summary_md, mass_plot, lift_plot, heatmap_plot]
        )
        heatmap_head.change(
            update_heatmap,
            inputs=[run_dropdown, heatmap_head],
            outputs=[heatmap_plot]
        )
        refresh_btn.click(
            lambda: gr.Dropdown(choices=get_run_names()),
            outputs=[run_dropdown]
        )

        # Initial load
        demo.load(
            lambda: (get_run_names()[0] if get_run_names() else None),
            outputs=[run_dropdown]
        ).then(
            update_demo,
            inputs=[run_dropdown],
            outputs=[summary_md, mass_plot, lift_plot, heatmap_plot]
        )

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()