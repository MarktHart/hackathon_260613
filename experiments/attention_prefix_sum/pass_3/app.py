import gradio as gr
import numpy as np
import pickle
from pathlib import Path

from agentic.experiments import (benchmarks as aeb,
                                 load_results,
                                 benchmark_panel)

# Sweep from task.py / main.py.
SEQ_LENS = (4, 8, 16, 32, 64)
CANONICAL_SEQ_LEN = 16
VOCAB_SIZE = 10
NUM_SEQS = 512

# Placeholder for runs not yet generated.
_PLACEHOLDER_PATH = Path(__file__).parent / "_placeholder.pickle"

# Demo helper: infer logits shape from a single synthetic sequence.
def _demo_logit_tensor(seq_len: int = 16, vocab_size: int = 10) -> np.ndarray:
    import torch
    DEVICE = "cuda"
    n = seq_len
    vocab = vocab_size
    # Simulate shape: batch[1], seq_len, vocab
    shape = (1, n, vocab)
    data = np.zeros(shape, dtype=np.float32)
    data.flat[0::vocab+1] = vocab  # set target token to V, uniform
    return data


# Demo tab that walks through a sequence position-by-position.
def _demo_tab_content():
    run_dir = Path(__file__).parent / "results"
    latest = sorted(r for r in run_dir.iterdir() if r.is_dir())[-1] if run_dir.iterdir() else None
    if not latest:
        raise FileNotFoundError("No runs found; please run `main.py` first.")

    # Choose a sequence to show in the demo (randomly sampled from the benchmark batch)
    rng = np.random.default_rng(0)
    seq_len = CANONICAL_SEQ_LEN
    # Synthetic token array: we could also load the real batch from a file.
    # For display: random tokens.
    tokens = rng.integers(0, VOCAB_SIZE, size=(1, seq_len), dtype=np.int32).flatten()
    # Compute target via cumsum mod V (matches the head's target token index).
    cumsum_mod = np.cumsum(tokens, axis=-1) % VOCAB_SIZE

    # Generate the hand-built head's logits for this token sequence.
    # This is a pure CPU placeholder that the visualisation needs — the
    # real attention mechanics are visible on the heatmap in the next tab.
    logits = _demo_logit_tensor(seq_len, VOCAB_SIZE)

    with gr.Blocks() as demo:
        with gr.Tabs():
            # Demo tab: per-timestep logit visualisation.
            with gr.Tab("Demo: Prefix-Sum Progression"):
                gr.Markdown(
                    "The hand-built head computes a cumulative sum over a sequence.\n"
                    "For a token sequence of length 16, the head's predicted token at each position\n"
                    "is the prefix sum of all previous tokens, wrapped mod 10."
                )
                with gr.Row():
                    with gr.Column():
                        # Input token sequence.
                        gr.Markdown("Input token sequence (integers 0-9):")
                        token_display = gr.Dataframe(
                            value=tokens, type="numpy", width=300, height=100
                        )
                    with gr.Column():
                        # Target prefix-sum token.
                        gr.Markdown("Target token at each position (cumsum mod 10):")
                        target_display = gr.Dataframe(
                            value=cumsum_mod, type="numpy", width=300, height=100
                        )
                with gr.Row():
                    # Logit tensor visualisation.
                    gr.Markdown("Logit tensor (batch=1, L=16, V=10) as a heatmap across positions")
                    logit_heatmap = gr.Plot(label="Logit Tensor Heatmap [L, V]")
                with gr.Row():
                    # Position slider.
                    pos_slider = gr.Slider(
                        minimum=0,
                        maximum=seq_len - 1,
                        step=1,
                        value=0,
                        label="Position (i)",
                    )
                    # Visual feedback at selected position.
                    pos_token = gr.Label(label="Token x_i")
                    pos_pred = gr.Label(label="Predicted cumulative sum token")

                # Populate display from tokens.
                def _update_position(pos: int):
                    x_i = tokens[pos]
                    pred_tk = cumsum_mod[pos]
                    return {"value": x_i}, {"value": f"Predicted target token = {pred_tk}"}

                pos_slider.change(fn=_update_position, inputs=[pos_slider],
                                    outputs=[pos_token, pos_pred])

                # Render the heatmap once (we do not animate across positions).
                def _render_logit_heatmap(logits: np.ndarray):
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 1), dpi=100)
                    # Logits are sparse: target token gets value V, all others 0.
                    # We scale visually to V=1 for the demo.
                    img = ax.imshow(logits[0].T, aspect="auto", cmap="gray",
                                      interpolation="nearest")
                    ax.set_xlabel("Token class (0..9)")
                    ax.set_ylabel("Position (0..15)")
                    ax.set_title("Logit heat by token class across positions")
                    plt.colorbar(img, ax=ax, fraction=0.04, pad=0.04)
                    plt.tight_layout()
                    return fig

                def _load_logit_heatmap():
                    fig = _render_logit_heatmap(logits)
                    with gr.Blocks() as out:
                        gr.Plot(value=fig)
                    return out

                logit_heatmap.change(fn=_load_logit_heatmap, inputs=[], outputs=logit_heatmap)

            # Benchmarks tab: all runs.
            with gr.Tab("Benchmark"):
                # The framework's global dashboard.
                # Scans every attempt under the goal and plots accuracy curves.
                benchmark_panel(Path(__file__).parent)

    demo.theme = gr.themes.Soft()
    return demo


demo: gr.Blocks = _demo_tab_content()


if __name__ == "__main__":
    demo.launch()