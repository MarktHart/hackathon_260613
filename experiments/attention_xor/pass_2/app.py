import gradio as gr
import numpy as np
from agentic.experiments import benchmark_panel

# This import pulls in the hand-built model from main, exposing `model_fn`.
from main import model_fn
from task import generate, CANONICAL_P, _fmt_p, _check_unit, SWEEP_PS

# Gradio Demo tab shows the XOR truth table and the hand-built logits.
with gr.Blocks() as demo:
    # ---- Title and description ----
    gr.Markdown("# attention_xor\n")
    gr.Markdown(
        "The model is `main.model_fn(tokens)` as printed in the commit diff.\n"
        "It uses **only the two token features** (A and B) and **no learned parameters**."
    )

    # ---- Interactive demo ----
    with gr.Tabs():
        with gr.Tab("XOR truth table"):
            # UI: two binary toggles, A and B.
            A_switch = gr.Checkbox(label="A = 1")
            B_switch = gr.Checkbox(label="B = 1")
            # UI: a table showing the four combinations and the model's logit.
            table = gr.Markdown(value="| A | B | XOR | Logit |\n|---|---|-----|-------|\n")

            def _render_table(a: bool, b: bool) -> str:
                # Build the token row that matches this selection.
                token_row = np.zeros((1, 4), dtype=np.int32)
                token_row[0, 1] = int(a) + 1  # A=0→1, A=1→2
                token_row[0, 2] = int(b) + 3  # B=0→3, B=1→4
                # Compute the hand-built logit.
                A = token_row[0, 1] - 1
                B = token_row[0, 2] - 3
                logit = (A - B) ** 2
                # XOR ground truth.
                xor_true = int(a != b)
                # Row entry.
                row = f"| {int(a)} | {int(b)} | {xor_true} | {logit:.4f} |\n"
                return table.value + row

            # Attach to both switches.
            A_switch.change(
                fn=_render_table,
                inputs=[A_switch, B_switch],
                outputs=table,
            )
            B_switch.change(
                fn=_render_table,
                inputs=[A_switch, B_switch],
                outputs=table,
            )

        with gr.Tab("Benchmark history"):
            # The shared benchmark panel that shows every attempt's metric scores.
            # It reads every attempt's `results/*/*/benchmark.json` under this goal.
            benchmark_panel("./..")

if __name__ == "__main__":
    demo.launch()