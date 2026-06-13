import gradio as gr
from agentic.experiments import load_task, benchmark_panel, results_dir
import pandas as pd
import torch
import numpy as np

# Load task (exposes meta like VOCAB_SIZE, etc.)
task = load_task(__file__)
VOCAB_SIZE = 32
SEQ_LEN = 16
N_BATCH = 1024

# -------------------------------------------------
# Demo Tab
# -------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown("""
    # Wildcard N-gram Attention demo

    This attempt implements a **single-attention-head transformer with a hand-coded
    pattern-matching matrix** that explicitly steers attention from the
    target token (id=2) back to the anchor token (id=1).

    The attention head sees a synthetic sequence:

    `[ 1 (anchor), w1, ..., wk, 2 (target), 0, ..., 0 (filler) ]`

    where `k` is the wildcard span (`k in [0,1,2,3,4]`).

    **Goal:** The head's attention matrix at `query=2` (the target) should peak
    at key position 0 (the anchor) across the wildcard span sweep, and should ignore
    the wildcard positions (ids 10-31).

    Below are two visual proofs:

    1. A **heatmap** of the attention weights from the target position to all previous positions,
       averaged over the entire batch.

    2. A **line chart** of `sharpness` (anchor weight / (wildcard + filler weights + ε)),
       which should remain high across all spans, indicating the pattern holds.

    Changing the wildcard span with the slider will reload a fresh run's data and update both views.
    """)

    # Controls
    with gr.Row():
        span_slider = gr.Slider(0, 4, step=1, value=1, label="Wildcard Span K")
        refresh_btn = gr.Button("Regenerate run")

    # Heatmap (left column)
    with gr.Column():
        heatmap_comp = gr.Plot(label="Target → All Keys Attention Heatmap")
    # Accuracy line (right column)
    with gr.Column():
        sharpness_comp = gr.Plot(label="Sharpness vs Wildcard Span")

    # Demo interaction functions
    def _load_latest_payload():
        run_dir = results_dir(__file__)
        import json
        path = f"{run_dir}/benchmark.json"
        with open(path) as f:
            payload = json.load(f)
        return payload

    def _attn_to_dataframe(attn: np.ndarray):
        B, L, _ = attn.shape
        # Mean attention across batch, then reshape to [L, L] for Altair
        mean_attn = attn[:, :L, :L].mean(0)
        df = pd.DataFrame({
            'query_pos': np.repeat(np.arange(L), L),
            'key_pos': np.tile(np.arange(L), L),
            'weight': mean_attn.ravel(),
        })
        df = df.drop_duplicates(subset=['query_pos', 'key_pos'])
        return df

    def _sweep_to_dataframe(sweep: list):
        recs = []
        for r in sweep:
            recs.append({
                "span": r["wildcard_span"],
                "sharpness": r["sharpness"],
            })
        return pd.DataFrame(recs)

    def _on_span_change(k: int):
        payload = _load_latest_payload()
        # Find the record for span k or the closest available
        sweep = payload["sweep"]
        rec = next((r for r in sweep if r["wildcard_span"] == k), sweep[-1])
        input_ids = rec["sequences"]
        assert input_ids.shape[-1] == SEQ_LEN, "Mismatch with expected sequence length"

        # Run the model function to compute attention for this batch
        batch = Batch(
            sequences=input_ids,
            anchor_pos=0,
            wildcard_pos=1,
            target_pos=1 + k,
            wildcard_span=k,
            anchor_token=1,
            target_token=2,
            wildcard_token_range=(10, 31),
        )

        attn = model_fn(batch)
        # Build attention heatmap data
        attn_df = _attn_to_dataframe(attn)

        # Update the line chart with the full sweep
        full_sweep_df = _sweep_to_dataframe(sweep)

        return [attn_df, full_sweep_df]

    span_slider.change(
        fn=_on_span_change,
        inputs=span_slider,
        outputs=[heatmap_comp, sharpness_comp],
    )

    # Placeholder for demo interaction
    refresh_btn.click(fn=lambda: None, inputs=[], outputs=[])

    # -------------------------------------------------
    # Benchmark Tab
    # -------------------------------------------------
    with gr.Tab("Benchmark"):
        benchmark_panel(__file__)

# Launch to satisfy boot-check
if __name__ == "__main__":
    demo.launch()