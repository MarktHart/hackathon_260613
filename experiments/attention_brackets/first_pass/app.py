import gradio as gr
from agentic.experiments import benchmark_panel

import json
from pathlib import Path

with gr.Blocks() as demo:
    # Demo tab: inspect one sequence's attention heatmap.
    with gr.Tabs() as tabs:
        with gr.TabItem("Demo"):
            with gr.Row():
                depth_input = gr.Number(label="Nesting depth (1–5)", value=3, interactive=True)
                refresh_btn = gr.Button("Refresh sequence")
                refresh_btn.click(
                    fn=None,
                    inputs=depth_input,
                    outputs=gr.empty(),
                )
            with gr.Row():
                with gr.Column():
                    seq = gr.Textbox(label="Bracket tokens (0=open, 1=close, 2=pad)")
            with gr.Row():
                attn_viz = gr.Image(
                    value=None,
                    label="Causal attention heatmap (L, L)",
                    width="960",
                    height="240",
                )
        with gr.TabItem("Benchmark"):
            # Reuse the shared benchmark panel: leaderboard and lift curves.
            # It automatically scans the goal's folder and includes the current
            # attempt.
            panel = benchmark_panel(Path(__file__).parent.parent)
            panel.render()

    # Populate demo with a random example on load.
    demo.load(
        fn=lambda: generate_random_sequence(),
        inputs=[],
        outputs=[seq, attn_viz],
    )


def generate_random_sequence():
    """Hand-built: pick one sequence at the canonical depth and return its
    tokens and the attention matrix from the hand-built head.
    """
    batch = load_batch()
    depth = batch.canonical_depth
    seqs = batch.sequences[depth]
    tokens = np.asarray(seqs[0], dtype=np.int32)  # first row

    # Compute attention like main.py (_naive_matching_head).
    L = tokens.shape[0]
    mask = np.tril(np.ones((L, L), dtype=np.float32))

    # True match array.
    true_match = np.zeros(L, dtype=np.float32)
    stack: list[int] = []
    for i, t in enumerate(tokens):
        if t == 0:
            stack.append(i)
        elif t == 1:
            true_match[i] = stack.pop()

    # Queries: closers get their matching opener; non-closers -1e12.
    q = np.zeros(L, dtype=np.float32)
    for i, t in enumerate(tokens):
        if t == 1:
            q[i] = true_match[i]
        else:
            q[i] = -1e12

    # Keys: opens get their index; others -1e12.
    k = np.zeros(L, dtype=np.float32)
    for i, t in enumerate(tokens):
        if t == 0:
            k[i] = i
        else:
            k[i] = -1e12

    attn = q[:, None] * k[None, :]
    attn = np.where(mask, attn, -np.inf)

    row_max = attn.max(axis=1, keepdims=True)
    row_max[row_max == -np.inf] = 0
    attn = np.exp(attn - row_max)

    norms = attn.sum(axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    attn = attn / norms

    # Return JSON serializable structures for the UI.
    return {
        "seq": tokens.tolist(),
        "attn": attn.tolist(),
    }


# Small stub to import the same batch used in `main.py`.
def load_batch():
    import experiments.attention_brackets.task as task
    return task.generate(seed=0)


if __name__ == "__main__":
    demo.launch()