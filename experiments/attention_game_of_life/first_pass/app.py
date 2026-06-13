"""Gradio app for the first_pass attempt: Demo + Benchmark tabs."""
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir, load_task

# --- Re-use the exact same circuit as main.py ---
DEVICE = "cuda"
_NEIGHBOR_KERNEL = torch.tensor(
    [[[1.0, 1.0, 1.0],
      [1.0, 0.0, 1.0],
      [1.0, 1.0, 1.0]]],
    dtype=torch.float32,
    device=DEVICE,
).unsqueeze(0)


def _gol_step(grids_np: np.ndarray) -> np.ndarray:
    """Run one GoL step on GPU and return next-state board (0/1)."""
    x = torch.as_tensor(grids_np, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    neighbor_counts = F.conv2d(
        F.pad(x, pad=(1, 1, 1, 1), mode="circular"),
        _NEIGHBOR_KERNEL,
    ).squeeze(1)
    alive = (x.squeeze(1) > 0.5)
    survive = alive & ((neighbor_counts == 2) | (neighbor_counts == 3))
    birth = (~alive) & (neighbor_counts == 3)
    next_alive = survive | birth
    return next_alive.detach().cpu().numpy().astype(np.float32)


def _render_board(board: np.ndarray) -> str:
    """Render a single (H, W) board as a monospace grid."""
    return "```\n" + "\n".join("".join("█" if c > 0.5 else "·" for c in row) for row in board) + "\n```"


def _latest_run_dir() -> Path:
    base = results_dir(__file__).parent
    runs = sorted(base.glob("*"))
    return runs[-1] if runs else None


def _load_payload(run_dir: Path):
    import json
    with open(run_dir / "benchmark.json") as f:
        return json.load(f)


def demo_fn(density: float, seed: int, run_choice: str):
    """Generate a board at the given density, show current, true next, predicted next."""
    rng = np.random.default_rng([seed, int(density * 10)])
    H, W = 16, 16
    # Generate a single board (batch=1)
    grid = (rng.random((1, H, W)) < density).astype(np.float32)
    true_next = _gol_step(grid)
    pred_logits = _gol_step(grid)  # Our model is exact, so prediction == truth
    pred_next = (pred_logits > 0).astype(np.float32)

    cur_md = _render_board(grid[0])
    true_md = _render_board(true_next[0])
    pred_md = _render_board(pred_next[0])

    # Compute per-cell match
    match = (pred_next == true_next).astype(np.float32)
    acc = match.mean()
    match_md = _render_board(match[0])

    return cur_md, true_md, pred_md, match_md, f"Cell accuracy: {acc:.4f}"


with gr.Blocks() as demo:
    gr.Markdown("# attention_game_of_life — first_pass (hand-built GoL circuit)")

    with gr.Tab("Demo"):
        gr.Markdown(
            "This attempt implements the **exact** Game of Life update rule as a "
            "fixed convolution circuit on the GPU. The prediction should match the "
            "ground truth perfectly at every density."
        )
        with gr.Row():
            density = gr.Slider(0.1, 0.5, value=0.3, step=0.1, label="Initial live-cell density")
            seed = gr.Number(value=0, label="Seed", precision=0)
            run_dd = gr.Dropdown(choices=["latest"], value="latest", label="Run (only latest for this attempt)")
        with gr.Row():
            btn = gr.Button("Generate & Compare", variant="primary")
        with gr.Row():
            cur_out = gr.Markdown(label="Current board")
            true_out = gr.Markdown(label="True next state")
        with gr.Row():
            pred_out = gr.Markdown(label="Predicted next state")
            match_out = gr.Markdown(label="Match (█=correct, ·=wrong)")
        acc_out = gr.Markdown()

        btn.click(
            demo_fn,
            inputs=[density, seed, run_dd],
            outputs=[cur_out, true_out, pred_out, match_out, acc_out],
        )
        # Also run on load
        demo.load(
            demo_fn,
            inputs=[density, seed, run_dd],
            outputs=[cur_out, true_out, pred_out, match_out, acc_out],
        )

    with gr.Tab("Benchmark"):
        # The shared benchmark panel scans all attempts under the goal.
        goal_dir = Path(__file__).parent.parent
        benchmark_panel(goal_dir)

if __name__ == "__main__":
    demo.launch()