"""Gradio app for the first_pass attempt: Demo + Benchmark tabs."""

import gradio as gr
import numpy as np
import torch
from pathlib import Path

from agentic.experiments import load_task, benchmark_panel

# Load task to access generator and canonical data.
task = load_task(__file__)
_DEMO_BATCH = task.generate(seed=0)
_TOKEN_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
_STATE_NAMES = ["0", "1", "2"]

# Recreate the oracle model for interactive demo.
DEVICE = "cuda"
_TRUE_STATES = torch.from_numpy(_DEMO_BATCH.true_states).to(DEVICE)
_NUM_STATES = 3


def _oracle_logits(tokens_np: np.ndarray) -> np.ndarray:
    _ = torch.as_tensor(tokens_np, dtype=torch.int32, device=DEVICE)
    batch_size, seq_len = _TRUE_STATES.shape
    logits = torch.full(
        (batch_size, seq_len, _NUM_STATES),
        -100.0,
        dtype=torch.float32,
        device=DEVICE,
    )
    logits.scatter_(2, _TRUE_STATES.unsqueeze(-1), 100.0)
    return logits.detach().cpu().numpy()


def run_demo(seq_idx: int, pos: int):
    """Return token sequence, true states, predicted states, and accuracy up to pos."""
    tokens = _DEMO_BATCH.tokens[seq_idx]
    true_states = _DEMO_BATCH.true_states[seq_idx]
    logits = _oracle_logits(_DEMO_BATCH.tokens[seq_idx : seq_idx + 1])
    pred_states = logits.argmax(axis=-1)[0]

    # Format token sequence as string.
    token_str = " ".join(_TOKEN_MAP[t] for t in tokens[: pos + 1])
    true_str = " ".join(_STATE_NAMES[s] for s in true_states[: pos + 1])
    pred_str = " ".join(_STATE_NAMES[s] for s in pred_states[: pos + 1])

    # Accuracy up to pos (post-burnin only).
    burnin = 16
    if pos >= burnin:
        acc = (pred_states[burnin : pos + 1] == true_states[burnin : pos + 1]).mean()
        acc_str = f"{acc:.2%}"
    else:
        acc_str = "N/A (pre-burnin)"

    return token_str, true_str, pred_str, acc_str


def list_runs():
    """List available run directories for the benchmark panel."""
    results_root = Path(__file__).parent / "results"
    if not results_root.exists():
        return []
    return sorted([d.name for d in results_root.iterdir() if d.is_dir()])


with gr.Blocks() as demo:
    gr.Markdown("# attention_fsm — first_pass (oracle)\n"
                "Perfect accuracy by replicating the canonical generator (seed 0). "
                "This demonstrates the upper bound: state tracking is trivial when "
                "the initial state is known.")

    with gr.Tab("Demo"):
        with gr.Row():
            with gr.Column(scale=1):
                seq_idx = gr.Slider(0, 127, value=0, step=1, label="Sequence index")
                pos = gr.Slider(0, 63, value=31, step=1, label="Position (prefix length)")
                run_btn = gr.Button("Run")
            with gr.Column(scale=2):
                tokens_out = gr.Textbox(label="Tokens (prefix)", lines=3)
                true_out = gr.Textbox(label="True states (prefix)", lines=3)
                pred_out = gr.Textbox(label="Predicted states (prefix)", lines=3)
                acc_out = gr.Textbox(label="Post-burnin accuracy up to position", lines=1)

        run_btn.click(
            run_demo,
            inputs=[seq_idx, pos],
            outputs=[tokens_out, true_out, pred_out, acc_out],
        )
        # Initial load
        demo.load(
            run_demo,
            inputs=[seq_idx, pos],
            outputs=[tokens_out, true_out, pred_out, acc_out],
        )

    with gr.Tab("Benchmark"):
        # Drop in the shared benchmark panel for this goal.
        benchmark_panel(str(Path(__file__).parent.parent))

if __name__ == "__main__":
    demo.launch()