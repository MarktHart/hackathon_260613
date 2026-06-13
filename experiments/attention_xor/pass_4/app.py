import gradio as gr
import numpy as np
from agentic.experiments import benchmark_panel

from main import main_model_fn

with gr.Blocks() as demo:
    gr.Markdown("# attention_xor")
    gr.Markdown("#### Pass 4 – single attention head solving XOR")
    gr.Markdown("A hand-built single attention head that embeds A and B into two orthogonal 2-d bases, pools the result, and returns a hand-coded XOR logit (`logit = 2*(A^2 - B^2)`). No MLP, no learnable parameters, fully deterministic on CUDA.")

    # Demo tab: render a single example
    with gr.Tabs():
        with gr.Tab("Single example demo"):
            # Two toggles to pick A and B values
            A_toggle = gr.Checkbox(label="A = 1")
            B_toggle = gr.Checkbox(label="B = 1")
            table_md = gr.Markdown(
                value="| A | B | XOR | Logit |\n"
                      "|---|---|-----|-------|\n"
            )
            acc_out = gr.Label(label="Demo accuracy for this example")

            def render_table(a: bool, b: bool) -> tuple[str, float]:
                # Build a single-token batch matching the toggles
                token_row = np.zeros((1, 4), dtype=np.int32)
                token_row[0, 1] = int(a) + 1   # 1-2 for A=0/1
                token_row[0, 2] = int(b) + 3   # 3-4 for B=0/1
                logits = main_model_fn(token_row)
                logit = float(logits[0])
                xor_truth = int(a != b)
                row = f"| {int(a)} | {int(b)} | {xor_truth} | {logit:.4f} |\n"
                new_table = table_md.value + row
                demo_acc = 1.0 if (logit > 0) == xor_truth else 0.0  # binary
                return new_table, demo_acc

            A_toggle.change(
                fn=render_table,
                inputs=[A_toggle, B_toggle],
                outputs=[table_md, acc_out],
            )
            B_toggle.change(
                fn=render_table,
                inputs=[A_toggle, B_toggle],
                outputs=[table_md, acc_out],
            )

        with gr.Tab("Benchmark history"):
            benchmark_panel("../../")

demo.load(renderer, inputs=[], outputs=[table_md, acc_out])

def initial_render():
    # default rendering of A=0, B=0
    return render_table(False, False)


if __name__ == "__main__":
    demo.launch()