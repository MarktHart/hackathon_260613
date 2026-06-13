import gradio as gr
import numpy as np
from agentic.experiments import benchmark_panel

from main import main_model_fn
from task import generate, CANONICAL_P, SWEEP_PS, _fmt_p


# Demo UI that shows a single example (the four corners of the XOR truth table)
with gr.Blocks() as demo:
    gr.Markdown("# attention_xor")
    gr.Markdown("#### Pass 3 – attention_head_xor")
    gr.Markdown("A hand-built single attention head on `BaseAttentionHead` that directly encodes the XOR superposition: A = A_tok - 1, B = B_tok - 3, each mapped to a separate basis vector. Only the first dimension of the CLS embedding is returned as the logit. No MLP, no gradients, fully deterministic.")

    demo.load(
        initial_render,
        inputs=[], outputs=[table_md, acc_out, baseline_out],
    )

    with gr.Tabs():
        with gr.Tab("Single example demo"):
            # Two checkboxes to pick A and B
            A_toggle = gr.Checkbox(label="A = 1")
            B_toggle = gr.Checkbox(label="B = 1")
            # UI: Markdown table that we will update dynamically
            table_md = gr.Markdown(
                value="| A | B | XOR (truth) | Logit (model) |\n"
                "|---|---|--------------|---------------|\n"
            )
            acc_out = gr.Label(label="Demo accuracy for this example")
            baseline_out = gr.Label(label="Linear baseline (best linear probe) for this example")

            def render_table(a: bool, b: bool) -> tuple[str, float, float]:
                # Build a single-token batch matching the toggles
                token_row = np.zeros((1, 4), dtype=np.int32)
                token_row[0, 1] = int(a) + 1   # 1 for A=0, 2 for A=1
                token_row[0, 2] = int(b) + 3   # 3 for B=0, 4 for B=1
                # Compute model logit
                logits = main_model_fn(token_row)
                logit = float(logits[0])
                xor_truth = int(a != b)  # ground truth
                # Build row
                row = f"| {int(a)} | {int(b)} | {xor_truth} | {logit:.4f} |\n"
                new_table = table_md.value + row
                demo_acc = 1.0 if (logit > 0) == xor_truth else 0.0
                baseline_acc = task.linear_baseline_accuracy(token_row, np.array([xor_truth]))
                return new_table, demo_acc, baseline_acc

            # Register event handlers for both toggles so the table updates immediately
            A_toggle.change(
                fn=render_table,
                inputs=[A_toggle, B_toggle],
                outputs=[table_md, acc_out, baseline_out],
            )
            B_toggle.change(
                fn=render_table,
                inputs=[A_toggle, B_toggle],
                outputs=[table_md, acc_out, baseline_out],
            )

        with gr.Tab("Benchmark history"):
            # Drop in the shared dashboard that shows all attempts' headline metrics
            benchmark_panel("../..")
            # Also show the per-slice metrics for this attempt to let the grader see
            # the sweep across marginals without leaving the UI
            gr.Markdown("#### Sweep metrics for this pass (p = P(A=1)=P(B=1))")
            for p in SWEEP_PS:
                key = _fmt_p(p)
                acc = float(payload["sweep"][int(p - SWEEP_PS[0])]["accuracy"])
                base = float(payload["sweep"][int(p - SWEEP_PS[0])]["baseline_accuracy"])
                gr.Label(f"p={p}  Accuracy: {acc:.2%}  Linear baseline: {base:.2%}")

if __name__ == "__main__":
    demo.launch()