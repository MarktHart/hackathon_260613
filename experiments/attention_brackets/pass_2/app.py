import gradio as gr
from agentic.experiments import benchmark_panel, load_task

import json
import numpy as np
from pathlib import Path

with gr.Blocks() as demo:
    # Demo tab: inspect one sequence's attention heatmap.
    with gr.Tabs() as tabs:
        with gr.TabItem("Demo"):
            with gr.Row():
                depth_input = gr.Number(
                    label="Nesting depth (1–5)", value=3, interactive=True, minimum=1, maximum=5
                )
                refresh_btn = gr.Button("Refresh sequence", variant="primary")
            with gr.Row():
                seq_display = gr.Textbox(label="Bracket tokens (0=open, 1=close, 2=pad)", interactive=False)
            with gr.Row():
                attn_viz = gr.Image(
                    format="png",
                    value=None,
                    label="Causal attention heatmap",
                    width=960,
                    height=240,
                    image_mode="RGB",
                    interactive=False,
                )

        with gr.TabItem("Benchmark"):
            # Reuse the shared benchmark panel from agentic.experiments.
            panel = benchmark_panel(Path(__file__).parent.parent)
            panel.render()

    # Helper to generate a heatmap image from an attention matrix.
    def _attn_to_image(attn):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(9.6, 2.4))
        im = ax.imshow(attn, cmap="viridis", origin="lower", aspect="auto")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        ax.grid(False)
        fig.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue(), fig.canvas.figure

    # Load and render a random example at the requested depth.
    # demo.load() must return a tuple (seq, attn_image) that the output
    # blocks expect.
    demo.load(
        fn=lambda: _demo_random_sequence(depth_input.value),
        inputs=[depth_input],
        outputs=[seq_display, attn_viz],
    )

    refresh_btn.click(
        fn=lambda: _demo_random_sequence(depth_input.value),
        inputs=[depth_input],
        outputs=[seq_display, attn_viz],
    )


# Small stub that loads the same batch used in main.py.
def _demo_random_sequence(depth: int):
    import experiments.attention_brackets.task as task
    batch = task.generate(seed=0)

    # Select a canonical sequence at the user's chosen depth (depth 3 by default).
    if depth not in batch.depths:
        raise ValueError(f"depth={depth} not available; batch.depths={batch.depths}")
    tokens, match = batch.sequences[depth][0], batch.matches[depth][0]

    # Compute attention like in main.py (_recursive_stack_head).
    model_fn = lambda t: _recursive_stack_head(np.asarray(t, dtype=np.int32))
    attn = model_fn(tokens)

    # Return JSON serializable structures.
    return {
        "seq": tokens.tolist(),
        "attn_image": _attn_to_image(attn)[0],  # the PNG byte buffer
    }


# Hand-implement the same attention routine as main.py so the demo can
# recompute a fresh matrix without a separate import.
def _recursive_stack_head(tokens: np.ndarray) -> np.ndarray:
    L = tokens.shape[0]
    mask = np.tril(np.ones((L, L), dtype=np.float32))

    # True match array (for reference only).
    true_match = np.zeros(L, dtype=np.int32)
    stack = []
    for i, t in enumerate(tokens):
        if t == 0:
            stack.append(i)
        elif t == 1:
            true_match[i] = stack.pop()

    # Initialise Query array.
    q = np.zeros((L, 1), dtype=np.float32)

    # State recurrence.
    state = []
    for i, t in enumerate(tokens):
        if t == 0:  # push opener
            state.append(np.full((1, 1), i, dtype=np.float32))
        elif t == 1 and state:  # peek top of stack
            q[i] = state[-1]
        else:
            q[i] = -1e12  # close with no opener; will be masked later

    # Key array: identity of position.
    k = np.arange(L).astype(np.float32)[:, None]  # (L, 1)

    # Score: closers emit their stack state as Query; K is position identity.
    raw_attn = q @ k.T  # (L, L)
    raw_attn = np.where(mask, raw_attn, -np.inf)

    row_max = raw_attn.max(axis=1, keepdims=True)
    exp_attn = np.exp(raw_attn - row_max)
    norms = exp_attn.sum(axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return exp_attn / norms


if __name__ == "__main__":
    demo.launch()

