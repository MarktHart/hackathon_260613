"""Gradio app for attention_reverse / first_pass.

Demo tab: pick a sequence length, see the hand-built head's attention pattern
(should be a clean anti-diagonal — query i -> key L-1-i), plus accuracy and
mirror-mass at that length. Benchmark tab: cross-attempt leaderboard/history.
"""
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel, load_task

GOAL_DIR = Path(__file__).resolve().parent.parent
ATTEMPT_DIR = Path(__file__).resolve().parent

task = load_task(__file__)
VOCAB = task.VOCAB_SIZE

# Import the hand-built model_fn factory from main.py.
import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("_rev_main", ATTEMPT_DIR / "main.py")
_main = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_main)


def _run(seq_len: int, beta: float):
    seq_len = int(seq_len)
    model_fn = _main.make_model_fn(VOCAB, beta=float(beta))
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, VOCAB, size=(8, seq_len), dtype=np.int64)
    logits, attn = model_fn(tokens)

    preds = logits.argmax(-1)
    targets = tokens[:, ::-1]
    acc = float((preds == targets).mean())

    mirror = np.arange(seq_len - 1, -1, -1)
    mass = float(attn[:, np.arange(seq_len), mirror].mean())

    # Attention heatmap (identical across batch — show [0]).
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(attn[0], cmap="magma", vmin=0, vmax=1)
    ax.plot(mirror, np.arange(seq_len), "c--", lw=1.5, alpha=0.7,
            label="mirror j = L-1-i")
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_title(f"attention pattern (L={seq_len})")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    sample = (
        f"input  : {tokens[0].tolist()}\n"
        f"pred   : {preds[0].tolist()}\n"
        f"reverse: {targets[0].tolist()}"
    )
    summary = (
        f"### L = {seq_len}\n"
        f"- **reverse accuracy**: {acc:.4f}\n"
        f"- **mirror attention mass**: {mass:.4f}\n"
        f"- random baseline: {1.0 / VOCAB:.4f}\n\n"
        f"```\n{sample}\n```"
    )
    return fig, summary


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_reverse — hand-built reversal head\n"
        "A single attention head with an exact discrete-Fourier mirror "
        "positional encoding: query `i` attends to key `L-1-i`. No learned "
        "weights; the construction is parametric in `L`, so it extrapolates to "
        "unseen lengths. A perfect head shows a clean **anti-diagonal** "
        "attention pattern and accuracy 1.0 at every length."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            len_dd = gr.Dropdown(
                choices=[int(s) for s in task.SEQ_LEN_SWEEP],
                value=int(task.CANONICAL_SEQ_LEN),
                label="sequence length",
            )
            beta_sl = gr.Slider(
                0.25, 8.0, value=1.0, step=0.25,
                label="softmax temperature (beta)",
            )
        plot = gr.Plot(label="attention pattern")
        info = gr.Markdown()

        len_dd.change(_run, inputs=[len_dd, beta_sl], outputs=[plot, info])
        beta_sl.change(_run, inputs=[len_dd, beta_sl], outputs=[plot, info])
        demo.load(_run, inputs=[len_dd, beta_sl], outputs=[plot, info])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
