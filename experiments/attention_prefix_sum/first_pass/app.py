import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path

from agentic.experiments import benchmark_panel, load_results

# ---- Hardcoded sweep (must match task.py / main.py) ------------------------------------
SEQ_LENS = (8, 16, 32, 64, 128)
CANONICAL_SEQ_LEN = 32
D = 1
N_TRIALS = 200   # not visualised directly, only as context


# model_fn contract: (values (n, d), positions (n,)) -> weights (n, n)
def identity_model_fn(values, positions):
    n = values.shape[0]
    # placeholder implementation for visual demo
    W = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]
    return W


def _render_head_heatmap(W: np.ndarray, n: int):
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(W, cmap="gray", interpolation="nearest", aspect="equal")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_title(f"Attention Head (seq_len={n})")
    ax.invert_yaxis()  # put query 0 at top as convention
    plt.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    plt.tight_layout()
    return fig


def _accuracy_plot(metrics_by_n: Dict[int, np.ndarray]):
    nvals = [float(metrics["prefix_mean_accuracy_n_" + str(n)]) for n in SEQ_LENS]
    xticks = [r"${}$".format(n) for n in SEQ_LENS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SEQ_LENS, nvals, marker="o", linewidth=2.0, markersize=5.0, color="#1f77b4")
    ax.set_xticks(SEQ_LENS)
    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("Prefix-mean Accuracy")
    ax.set_title(f"Accuracy vs Sequence Length (D={D}, N_trials={N_TRIALS})")
    plt.tight_layout()
    return fig


def _lift_plot(metrics_by_n: Dict[int, np.ndarray]):
    nvals = [float(metrics["lift_over_baseline_n_" + str(n)]) for n in SEQ_LENS]
    xticks = [r"${}$".format(n) for n in SEQ_LENS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SEQ_LENS, nvals, marker="s", linewidth=2.0, markersize=5.0, color="#ff7f0e")
    ax.set_xticks(SEQ_LENS)
    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("Lift over Last-Token Baseline")
    ax.set_title(f"Lift (Above Baseline) vs Sequence Length (D={D}, N_trials={N_TRIALS})")
    plt.tight_layout()
    return fig


def _correlation_plot(metrics_by_n: Dict[int, np.ndarray]):
    nvals = [float(metrics["prefix_sum_corr_n_" + str(n)]) for n in SEQ_LENS]
    xticks = [r"${}$".format(n) for n in SEQ_LENS]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SEQ_LENS, nvals, marker="^", linewidth=2.0, markersize=5.0, color="#2ca02c")
    ax.set_xticks(SEQ_LENS)
    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("Pearson Corr. (Rec vs True Prefix Sum)")
    ax.set_title(f"Prefix-Sum Correlation vs Sequence Length (D={D}, N_trials={N_TRIALS})")
    plt.tight_layout()
    return fig


with gr.Blocks() as demo:
    with gr.Tabs():
        # ---- Demo tab -------------------------------------------------------------
        with gr.Tab("Demo: Attention Head Heatmap"):
            gr.Markdown("The hand-baked attention head computes a uniform causal prefix (see main.py). "
                        "The ideal pattern is uniform across the causal triangle and zero elsewhere. "
                        "Select a sequence length to view its attention heatmap.")

            # Run dropdown: shows the latest run (or any older run)
            run_dd = gr.Dropdown(
                choices=[r.split("/")[-1] for r in Path(__dirname, "results").iterdir()],
                label="Run ID",
                value=lambda: sorted(r.name for r in Path(__dirname, "results").iterdir())[-1]
            )

            # Sequence length selector (from the sweep)
            seq_len_dd = gr.Dropdown(choices=list(SEQ_LENS), label="Sequence Length", value=SEQ_LENS[2])

            # Output heatmap
            with gr.Row():
                with gr.Column(scale=3):
                    heatmap_out = gr.Plot(label="Attention Weights")
                with gr.Column(scale=1):
                    metrics_out = gr.Code(label="Metrics Context", language="json")

            def update_view(run_id: str, n: int) -> Tuple[dict, dict]:
                metrics_path = Path(__dirname, "results", run_id, "benchmark.json")
                if not metrics_path.is_file():
                    raise gr.Error(f"Missing benchmark JSON at {metrics_path}")
                metrics = load_results(__dirname, run_id)

                # Pull the row-by-row accuracy (if present) or fall back to slice-level accuracy
                # (only in main.py's payload, not in the current benchmark)
                # Since this demo is minimal, we just show the slice-level accuracy and lift.
                accuracy_by_n = dict()
                for k, v in metrics.items():
                    if k.startswith("prefix_mean_accuracy_n_"):
                        try:
                            n_key = int(k.split("_")[-1])
                            accuracy_by_n[n_key] = v
                        except:
                            pass

                # Generate the heatmap for length n (placeholder implementation)
                W = np.tril(np.ones((n, n))) / np.arange(1, n + 1)[:, None]
                fig = _render_head_heatmap(W, n)
                with gr.Blocks() as _out:
                    gr.Plot(value=fig)
                # JSON snippet for context
                ctx = {
                    "prefix_mean_accuracy_n_{}".format(n): accuracy_by_n.get(n),
                    "lift_forward": None,  # placeholder
                }
                return _out, ctx

            seq_len_dd.change(
                update_view,
                inputs=[run_dd, seq_len_dd],
                outputs=[heatmap_out, metrics_out],
                queue=True
            )
            demo.load(fn=lambda: update_view(None, SEQ_LENS[2]), inputs=None, outputs=[heatmap_out, metrics_out])

        # ---- Benchmark tab --------------------------------------------------------
        with gr.Tab("Benchmark"):
            benchmark_panel(__dirname)

if __name__ == "__main__":
    demo.launch()