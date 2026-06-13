import gradio as gr
import numpy as np
import json
from pathlib import Path
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent


def load_latest_run():
    """Find the most recent run directory and load its payload and benchmark."""
    results_base = Path(__file__).parent / "results"
    if not results_base.exists():
        return None, None, None
    run_dirs = sorted([d for d in results_base.iterdir() if d.is_dir()])
    if not run_dirs:
        return None, None, None
    latest = run_dirs[-1]
    # Find payload (we need to reconstruct from saved data or re-run? 
    # The task only saves benchmark.json. For demo we'll need the full payload.
    # Let's save the full payload in main.py too, but for now we'll load what we can.)
    benchmark_path = latest / "benchmark.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark = json.load(f)
    else:
        benchmark = None
    return latest, None, benchmark


def load_run(run_name):
    """Load a specific run by directory name."""
    results_base = Path(__file__).parent / "results"
    run_dir = results_base / run_name
    if not run_dir.exists():
        return None, None, None
    benchmark_path = run_dir / "benchmark.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            benchmark = json.load(f)
    else:
        benchmark = None
    return run_dir, None, benchmark


def plot_orthogonality_heatmap(q_proj, factors, factor_names=None):
    """Create a heatmap of encoding direction cosines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = q_proj.shape[0]
    from experiments.attention_lis.benchmark import _encoding_dirs
    W = _encoding_dirs(q_proj, factors)  # (K, K)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    W_norm = W / norms
    cosines = W_norm @ W_norm.T  # (K, K)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cosines, vmin=-1, vmax=1, cmap="RdBu_r", aspect="equal")
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    if factor_names:
        ax.set_xticklabels(factor_names)
        ax.set_yticklabels(factor_names)
    else:
        ax.set_xticklabels([f"Factor {i}" for i in range(K)])
        ax.set_yticklabels([f"Factor {i}" for i in range(K)])
    ax.set_title("Encoding Direction Cosine Similarity")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # Annotate values
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cosines[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(cosines[i, j]) > 0.5 else "black", fontsize=10)
    plt.tight_layout()
    return fig


def plot_sweep_curves(sweep, factors):
    """Plot orthogonality and alignment across noise sweep."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.attention_lis.benchmark import _orthogonality, _alignment

    noise_stds = [entry["noise_std"] for entry in sweep]
    orthos = [_orthogonality(entry["q_proj"], factors) for entry in sweep]
    aligns = [_alignment(entry["q_proj"], factors) for entry in sweep]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(noise_stds, orthos, "o-", label="Orthogonality", color="tab:blue")
    ax1.set_xlabel("Noise std")
    ax1.set_ylabel("Orthogonality (1 - |cos|)")
    ax1.set_title("LIS Orthogonality vs Noise")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(noise_stds, aligns, "o-", label="Alignment", color="tab:orange")
    ax2.set_xlabel("Noise std")
    ax2.set_ylabel("Mean correlation with factor")
    ax2.set_title("Factor Alignment vs Noise")
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    return fig


def plot_factor_projections(q_proj, factors):
    """Plot projected query values colored by factor value for each factor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = q_proj.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(4 * K, 4), squeeze=False)
    for k in range(K):
        ax = axes[0, k]
        proj_k = q_proj[k]  # (L,)
        factor_k = factors[:, k]
        pos = factor_k > 0
        neg = ~pos
        ax.hist(proj_k[pos], bins=20, alpha=0.6, label="factor=+1", color="tab:blue", density=True)
        ax.hist(proj_k[neg], bins=20, alpha=0.6, label="factor=-1", color="tab:red", density=True)
        ax.set_xlabel("Projection value")
        ax.set_ylabel("Density")
        ax.set_title(f"Factor {k} separation")
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def create_demo_tab(run_dir, payload, benchmark):
    """Create the Demo tab content for a given run."""
    if payload is None:
        # We don't have the full payload saved, only benchmark.json
        # Show benchmark metrics
        with gr.Group():
            gr.Markdown("## Benchmark Metrics (Latest Run)")
            if benchmark:
                for key, val in benchmark.items():
                    if key != "version":
                        gr.Markdown(f"- **{key}**: {val:.4f}" if isinstance(val, float) else f"- **{key}**: {val}")
            else:
                gr.Markdown("No benchmark data available.")
        return

    # Full payload available - create visualizations
    canonical = payload["canonical"]
    sweep = payload["sweep"]
    factors = payload["factors"]
    q_proj_canon = canonical["q_proj"]
    k_proj_canon = canonical["k_proj"]

    with gr.Group():
        gr.Markdown("## Canonical Condition (noise_std=0.1)")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Query Projections - Encoding Direction Cosines")
                plot_q = plot_orthogonality_heatmap(q_proj_canon, factors)
                gr.Plot(value=plot_q)
            with gr.Column():
                gr.Markdown("### Key Projections - Encoding Direction Cosines")
                plot_k = plot_orthogonality_heatmap(k_proj_canon, factors)
                gr.Plot(value=plot_k)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Query Projections - Factor Separation")
                plot_q_sep = plot_factor_projections(q_proj_canon, factors)
                gr.Plot(value=plot_q_sep)
            with gr.Column():
                gr.Markdown("### Key Projections - Factor Separation")
                plot_k_sep = plot_factor_projections(k_proj_canon, factors)
                gr.Plot(value=plot_k_sep)

    with gr.Group():
        gr.Markdown("## Robustness Sweep")
        sweep_plot = plot_sweep_curves(sweep, factors)
        gr.Plot(value=sweep_plot)

    with gr.Group():
        gr.Markdown("## Benchmark Metrics")
        if benchmark:
            for key, val in benchmark.items():
                if key != "version":
                    gr.Markdown(f"- **{key}**: {val:.4f}" if isinstance(val, float) else f"- **{key}**: {val}")


def get_run_choices():
    """Get list of available run directories."""
    results_base = Path(__file__).parent / "results"
    if not results_base.exists():
        return []
    run_dirs = sorted([d.name for d in results_base.iterdir() if d.is_dir()], reverse=True)
    return run_dirs


with gr.Blocks() as demo:
    gr.Markdown("# attention_lis — Linear Independent Subspaces in Attention")
    gr.Markdown("**Attempt:** `first_pass` (hand-built orthogonal subspaces)")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            run_dropdown = gr.Dropdown(
                choices=get_run_choices(),
                label="Select run",
                value=None,
                interactive=True,
            )
            load_btn = gr.Button("Load Run", variant="primary")

            demo_output = gr.HTML()
            # We'll use a hidden JSON component to pass data
            run_dir_state = gr.State(None)
            payload_state = gr.State(None)
            benchmark_state = gr.State(None)

            def on_load_run(run_name):
                if not run_name:
                    return None, None, None, "<p>No run selected</p>"
                run_dir, payload, benchmark = load_run(run_name)
                # Since we don't save full payload, we'll reconstruct minimal viz from benchmark
                # For now, show benchmark metrics
                html = "<div style='padding: 10px;'>"
                html += f"<h3>Run: {run_name}</h3>"
                if benchmark:
                    html += "<table style='width:100%; border-collapse:collapse;'>"
                    html += "<tr><th style='text-align:left; padding:5px; border-bottom:1px solid #ddd'>Metric</th><th style='text-align:right; padding:5px; border-bottom:1px solid #ddd'>Value</th></tr>"
                    for key, val in benchmark.items():
                        if key != "version":
                            html += f"<tr><td style='padding:5px; border-bottom:1px solid #eee'>{key}</td><td style='text-align:right; padding:5px; border-bottom:1px solid #eee'>{val:.4f}</td></tr>" if isinstance(val, float) else f"<tr><td style='padding:5px; border-bottom:1px solid #eee'>{key}</td><td style='text-align:right; padding:5px; border-bottom:1px solid #eee'>{val}</td></tr>"
                    html += "</table>"
                else:
                    html += "<p>No benchmark data</p>"
                html += "</div>"
                return run_dir, payload, benchmark, html

            load_btn.click(
                fn=on_load_run,
                inputs=[run_dropdown],
                outputs=[run_dir_state, payload_state, benchmark_state, demo_output],
            )

            # Auto-load latest on startup
            demo.load(
                fn=lambda: on_load_run(get_run_choices()[0]) if get_run_choices() else (None, None, None, "<p>No runs found</p>"),
                inputs=[],
                outputs=[run_dir_state, payload_state, benchmark_state, demo_output],
            )

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-Attempt Benchmark History")
            gr.Markdown("Leaderboard and metric trends across all attempts for this goal.")
            # The benchmark_panel function adds its own components to the current Blocks context
            benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()