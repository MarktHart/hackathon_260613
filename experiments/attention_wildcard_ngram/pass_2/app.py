import gradio as gr
from agentic.experiments import load_task, benchmark_panel, results_dir
import numpy as np
import torch

# -------------------------------------------------
# app.py
# Gradio Blocks with a Demo tab and Benchmark tab.
# -------------------------------------------------
with gr.Blocks() as demo:
    # Demo Tab
    with gr.Tab("Demo"):
        gr.Markdown("""
        # Wildcard N-gram Visualisation

        This attempt implements a **pattern-matching attention circuit**:
        - The model sees a synthetic sequence of the form
          `[ A w1..wk B C ]  [FILLER]  [ A w2..wk B ]`.
        - The query `A * B` must predict the continuation `C` copied from
          the earlier definition `A * B C`.
        - Wildcards `w1..wk` and `w2..wk` are drawn independently, so an exact
          n-gram match is almost never possible — only a true wildcard matcher
          should succeed.

        Below the heatmap is an animated scatter of accuracy per span `k`.
        The goal is to see the headlight-shaped curve that peaks at span 1 and
        tapers as the wildcard grows — the same shape the model would show
        if it really matched `A * B -> C`.
        """)

        # Controls
        with gr.Row():
            with gr.Column():
                demo_span_dd = gr.Slider(minimum=1, maximum=4, step=1,
                                         value=1, label="Wildcard Span")
            with gr.Column():
                demo_button = gr.Button("Regenerate")

        # Visuals
        with gr.Row():
            with gr.Column():
                # Attention heatmaps from the query B to all positions in the sequence
                heatmap_comp = gr.Plot(label="Attention Weights: from last B to all positions")
            with gr.Column():
                # Accuracy line per span
                acc_plot_comp = gr.Plot(label="Accuracy by Wildcard Span")

        # Helper to plot attention for a given batch position
        def _plot_attn_head_attn_at_q(a: torch.Tensor):
            b, l, c = a.shape
            # Sum attention across heads, normalize per row
            attn = a.mean(dim=1)  # shape (b, l, l)
            # Average over batch rows to give a mean attention image
            attn = attn.mean(dim=0)  # (l, l)

            # Build a pandas DataFrame compatible with Altair
            import pandas as pd
            df = pd.DataFrame({
                'key_pos': [i for i in range(attn.shape[1])],
                'query_pos': [j for j in range(attn.shape[0])],
                'weight': attn.ravel(),
            })
            return df

        def _plot_acc_curve(payloads):
            import pandas as pd
            data = []
            for payload in payloads:
                for rec in payload["sweep"]:
                    k = rec["wildcard_span"]
                    acc = rec["accuracy"]
                    data.append({"span": k, "accuracy": acc})
            df = pd.DataFrame(data)
            return df

        # Demo interaction
        def on_span_change(span: int):
            run_path = results_dir(__file__)
            # Load the most recent run's benchmark JSON
            import json
            with open(f"{run_path}/benchmark.json") as f:
                payload = json.load(f)
            sweeps = payload["sweep"]
            rec = next(r for r in sweeps if r["wildcard_span"] == span)
            input_ids = np.asarray(rec["input_ids"])
            # Compute attention weights for the current batch position
            attn_weights = WildcardNgramModel(task.vocab_size).to(device)(
                torch.from_numpy(input_ids).long()
            ).detach().cpu().numpy()
            # Convert model output to gr.Plot
            attn_df = _plot_attn_head_attn_at_q(attn_weights)
            acc曲线_df = _plot_acc_curve([payload])
            return [attn_df, acc曲线_df]

        demo_span_dd.change(fn=on_span_change, inputs=demo_span_dd, outputs=[heatmap_comp, acc_plot_comp])
        demo_button.click(fn=lambda: None, inputs=[], outputs=[])  # placeholder to make interactive

    # Benchmark Tab
    with gr.Tab("Benchmark"):
        benchmark_panel(__file__)

# Launch after construction to avoid event registration leaks
if __name__ == "__main__":
    demo.launch()