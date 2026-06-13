import gradio as gr
import numpy as np
import math
from typing import Any

# agentic imports.
from agentic.experiments import (
    generate as _task_generate,
    benchmark_panel,
    results_dir,
    load_task,
)

# Load the goal's task module to share constants.
from experiments.attention_argmax.task import generate as _task_generate_single

# ---- Demo tab: visualize one batch at canonical separation ----------
_N = 32   # keys/positions


def _make_batches_for_demo(num_batches: int = 1):
    """Generate one deterministic batch (canonical seed 0) and repeat to get num_batches."""
    # The goal's canonical generation.
    batch = _task_generate_single(0)   # seed=0 at canonical separation=2.0
    # For demo convenience, wrap in a list of B batches where all batches are identical.
    q = np.tile(batch.q[None, :], (num_batches, 1))          # (B, d)
    K = np.tile(batch.K[None, :, :], (num_batches, 1, 1))    # (B, N, d)
    V = np.tile(batch.V[None, :, :], (num_batches, 1, 1))    # (B, N, d)
    winner_idxs = np.full(num_batches, batch.winner_idx, dtype=np.int64)  # per-sequence winner
    # Model function we share with main: must match signature of task.evaluate.
    def model_fn(qi: np.ndarray, Ki: np.ndarray, Vi: np.ndarray = None):
        # Inside demo, we need a batched implementation that returns (B, N).
        B = q.shape[0]
        attn = np.zeros((B, _N), dtype=np.float32)
        tau = 1.0   # sharp, canonical attention
        for b in range(B):
            sims = np.dot(Ki[b], qi[b])   # (N,)
            sims -= npamax(sims)          # numerical stability
            exp sims = np.exp(sims / tau)
            attn[b] = exp_sims / np.sum(exp_sims)
        return attn
    # Run it now and cache.
    attn_weights = model_fn(q, K, V)
    # Verify normalization.
    np.testing.assert_almost_equal(attn_weights.sum(axis=1), 1.0, decimal=5)
    return q, K, V, winner_idxs, attn_weights


def _plot_attn(attn_weights: np.ndarray, winner_idxs: np.ndarray, seq_idx: int = 0) -> gr.Line:
    """Return a LinePlot showing attn_weights[seq_idx] with a red vertical at the winner."""
    y = attn_weights[seq_idx]
    x = np.arange(_N)
    # Find the index where attention has its single peak (approx)
    peak_idx = int(np.argmax(y))
    # Draw the peak with a red marker (if it aligns with the ground-truth winner, great)
    y[peak_idx] += 0.05  # tiny bump for visibility
    y = y.clip(0.0, 1.05)   # clip for demo range
    return gr.LinePlot(
        x=x,
        y=y,
        x_title="Key position (0–31)",
        y_title="Attention probability",
        title=f"Attention head (B={attn_weights.shape[0]})",
        width=500,
        height=250
    )


# ---- Gradio Demo ---------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("# attention_argmax: Demo of the hand-implemented argmax head")
    gr.Markdown("The head receives `q`, `K`, and `V` vectors and outputs a probability distribution over 32 positions. "
                "It places nearly all mass on the single key with the highest similarity (`i*`).")

    # 1. Visualise the batch (single batch, repeated B times for demo)
    with gr.Blocks():
        batch_view = gr.DataFrame(label="Batch", visible=False)

    # 2. Plot attention distribution for the selected sequence
    with gr.Blocks():
        seq_idx = gr.Slider(0, _N - 1, value=0, label="Sequence index")
        attn_plot = gr.Line(label="Attention probabilities", visible=False)

    # ---- On load: generate the demo batch and initialise plots ----
    def _on_load():
        q, K, V, winner_idxs, attn_weights = _make_batches_for_demo(B=128)
        # batch_view is a Dataframe; construct rows as {"q": "...", "K": "..."} for simplicity in demo
        # but we won't surface this since batch_view is hidden.
        # Instead, we just store the tensors for the plot callback.
        # For this demo, we don't expose the tensors — just plot the attention.
        return q, K, V, winner_idxs, attn_weights

    # ---- Demo lifecycle -------------------------------------------------
    demo.load(fn=_on_load, outputs=[attn_plot])

    # ---- Sequence slider callback ---------------------------------------
    def _on_seq_changed(seq_idx: int, attn_weights: np.ndarray, winner_idxs: np.ndarray):
        # Draw the vertical line at the winner position (red)
        if seq_idx < len(winner_idxs):
            x = np.arange(_N)
            y = attn_weights[seq_idx]
            # Add a tiny bump at the winner for visibility
            winner_pos = winner_idxs[seq_idx]
            y = y.copy()
            y[winner_pos] += 0.05
            return gr.LinePlot(x, y, x_title="Position", y_title="Probability", width=500, height=250)
        raise ValueError(f"Sequence index {seq_idx} out of bounds")

    seq_idx.change(
        fn=_on_seq_changed,
        inputs=[seq_idx, gr.State(_make_batches_for_demo(B=128)[4]), gr.State(_make_batches_for_demo(B=128)[3])],
        outputs=attn_plot,
    )
    demo.load(fn=_on_seq_changed, inputs=seq_idx, outputs=attn_plot)

    # ---- Benchmark tab: drop in leaderboard & metric history ----------
    with gr.Blocks():
        gr.Markdown("## Benchmark")
        gr.Markdown("### Leaderboard (Headline metric: `argmax_fidelity_canonical`)")
        benchmark_panel(_task_generate)

if __name__ == "__main__":
    demo.launch()