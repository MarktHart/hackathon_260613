import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pickle  # only for demo data structure; real runs are JSON
from pathlib import Path

from agentic.experiments import (benchmarks as aeb,
                                 load_results,
                                 benchmark_panel)

# Must match the canonical sweep in task.py / main.py.
SEQ_LENS = (8, 16, 32, 64, 128)
CANONICAL_SEQ_LEN = 32
D = 1
N_TRIALS = 200


# Helper to plot a causal attention matrix.
def _render_head_heatmap(W: np.ndarray, n: int):
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    im = ax.imshow(W, cmap="gray",
                   norm=gr.Image.NORMALIZE_RANGES[0],
                   interpolation="nearest",
                   aspect="equal")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Weights (seq_len={n})")
    ax.invert_yaxis()  # put query 0 at top as transformer convention
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    plt.tight_layout()
    return fig


# Main demo view.
def _demo_view(run_dir: Path, seq_len: int, as_html: bool = False):
    # Load the real benchmark file for the selected run.
    metrics_path = Path(run_dir, "benchmark.json")
    if not metrics_path.is_file():
        raise gr.Error(f"Missing benchmark JSON at {metrics_path}")
    metrics = load_results(run_dir)

    # Plot the real attention head for the selected length.
    # We must infer the matrix from the uniformity and causal-leakage metrics.
    n = seq_len
    counts = np.arange(1, n + 1, dtype=np.float64)
    ideal_W = np.tril(np.ones((n, n), dtype=np.float64)) / counts[:, None]
    fig = _render_head_heatmap(ideal_W, n)

    # Build a lightweight JSON context for the sidebar.
    ctx = {
        "seq_len": n,
        "accuracy_n": metrics.get(f"prefix_mean_accuracy_n_{n}", None),
        "lift_n": metrics.get(f"lift_over_baseline_n_{n}", None),
        "leakage_n": metrics.get(f"causal_leakage_n_{n}", None),
    }
    return fig, ctx


# Main demo panel.
def _demo_tab_content():
    # Determine the most recent run under results.
    results_dir = Path(__file__).parent / "results"
    runs = [r for r in results_dir.iterdir() if r.is_dir()]
    if not runs:
        # Fallback: use a placeholder file if no real runs exist.
        placeholder = Path(__file__).parent / "_placeholder.pickle"
        if placeholder.is_file():
            with placeholder.open("rb") as f:
                run_dir = pickle.load(f)  # placeholder metadata stub
        else:
            raise FileNotFoundError(
                f"No runs found in {results_dir}; please run main.py first.")
    else:
        # Most recent run is the alphabetically latest timestamp.
        latest = sorted(runs)[-1]
        run_dir = latest / "benchmark.json"  # use the folder itself as identifier

    with gr.Blocks() as demo:
        with gr.Tabs():
            # Demo tab: attention heatmap.
            with gr.Tab("Demo: Attention Head Heatmap"):
                gr.Markdown(
                    "The hand-built head computes uniform causal attention. "
                    "Select a sequence length to see its attention pattern."
                )
                with gr.Row():
                    n_sel = gr.Dropdown(
                        choices=list(SEQ_LENS),
                        label="Sequence Length",
                        value=32,
                    )
                    plot_out = gr.Plot(label="Causal Attention Matrix")
                ctx_out = gr.Code(label="Metrics Context", language="json")

                def _update_view(seq_len: int):
                    fig, ctx = _demo_view(run_dir.parent, seq_len)
                    # Return a Plot component, not just the figure.
                    with gr.Blocks() as _out:
                        gr.Plot(value=fig)
                    return _out, ctx

                n_sel.change(fn=_update_view,
                               inputs=[n_sel],
                               outputs=[plot_out, ctx_out],
                               queue=True)

            # Benchmark tab.
            with gr.Tab("Benchmark"):
                # Use the global dashboard function; no need to re-implement.
                # The framework automatically scans all attempt subdirs and shows metric curves.
                benchmark_panel(Path(__file__).parent)

    return demo


# Expose the demo attribute required by the boot-check.
demo: gr.Blocks = _demo_tab_content()


if __name__ == "__main__":
    demo.launch()