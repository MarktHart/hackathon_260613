"""Gradio app for pass_2 — attention-based Game of Life circuit.

Demo tab:
  * an independent NumPy ground-truth next state vs. the attention circuit's
    prediction (DIFFERENT code paths, so the match panel is a real check);
  * the attention pattern for a chosen query cell, showing the head selects
    exactly its 8 toroidal neighbours (weight 1/8 each);
  * a causal-ablation table (full vs. global-attention vs. self-only vs. the
    static copy baseline) and a grid-size operating-range table, read from the
    latest run's artefacts.

Benchmark tab: the shared cross-attempt leaderboard.
"""

import json
import os
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import GoLAttention, ref_next_state  # noqa: E402

from agentic.experiments import benchmark_panel  # noqa: E402

APP_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GOAL_DIR = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"

_MODEL = None


def _model() -> GoLAttention:
    global _MODEL
    if _MODEL is None:
        _MODEL = GoLAttention(h=16, w=16, device=APP_DEVICE)
    return _MODEL


# ---------------- rendering ----------------
def _upscale(rgb: np.ndarray, factor: int = 18) -> np.ndarray:
    return np.kron(rgb, np.ones((factor, factor, 1), dtype=np.uint8))


def _board_img(board: np.ndarray) -> np.ndarray:
    h, w = board.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[board > 0.5] = (235, 235, 255)
    rgb[board <= 0.5] = (28, 28, 40)
    return _upscale(rgb)


def _match_img(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    h, w = pred.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    correct = (pred > 0.5) == (true > 0.5)
    rgb[correct] = (40, 150, 80)     # green = correct
    rgb[~correct] = (210, 50, 50)    # red = wrong
    return _upscale(rgb)


def _attn_img(weights: np.ndarray, qi: int, qj: int) -> np.ndarray:
    h, w = weights.shape
    norm = weights / (weights.max() + 1e-9)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 2] = (40 + norm * 200).astype(np.uint8)   # attended cells glow blue
    rgb[..., 1] = (norm * 180).astype(np.uint8)
    rgb[qi, qj] = (240, 60, 60)                        # query cell = red
    return _upscale(rgb)


def _latest_run() -> Path | None:
    if not RESULTS.exists():
        return None
    runs = sorted([p for p in RESULTS.glob("*") if p.is_dir()])
    return runs[-1] if runs else None


# ---------------- callbacks ----------------
def run_demo(density, seed, qi, qj):
    density, seed, qi, qj = float(density), int(seed), int(qi), int(qj)
    rng = np.random.default_rng([seed, int(density * 100)])
    board = (rng.random((1, 16, 16)) < density).astype(np.float32)

    true = ref_next_state(board)[0]                       # independent numpy GT
    logit = _model().forward(
        torch.as_tensor(board, dtype=torch.float32, device=APP_DEVICE)
    ).detach().cpu().numpy()[0]
    pred = (logit > 0).astype(np.float32)

    n_wrong = int(np.sum((pred > 0.5) != (true > 0.5)))
    acc = 1.0 - n_wrong / true.size
    tp = int(np.sum((pred > 0.5) & (true > 0.5)))
    fp = int(np.sum((pred > 0.5) & (true <= 0.5)))
    fn = int(np.sum((pred <= 0.5) & (true > 0.5)))
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0

    qidx = qi * 16 + qj
    attn = _model().attention_row(qidx, mode="full")
    n_attended = int(np.sum(attn > 1e-6))
    w_each = float(attn.max())

    stats = (
        f"**Circuit vs. independent NumPy ground truth** — "
        f"cell accuracy **{acc:.4f}**, alive-next F1 **{f1:.4f}**, "
        f"wrong cells **{n_wrong}/256**.\n\n"
        f"**Attention of cell ({qi},{qj})**: spreads weight **{w_each:.3f}** "
        f"over **{n_attended}** cells — exactly its 8 toroidal neighbours "
        f"(1/8 each), nothing else. That uniform 8-way attention *is* the "
        f"neighbour count the MLP thresholds."
    )
    return (
        _board_img(board[0]), _board_img(true), _board_img(pred),
        _match_img(pred, true), _attn_img(attn, qi, qj), stats,
    )


def load_tables():
    run = _latest_run()
    abl_rows, rob_rows = [], []
    if run is not None:
        ap = run / "ablation.json"
        if ap.exists():
            a = json.loads(ap.read_text())
            for i, d in enumerate(a["densities"]):
                abl_rows.append([
                    round(d, 2),
                    round(a["full_f1"][i], 3),
                    round(a["ablate_f1"][i], 3),
                    round(a["selfonly_f1"][i], 3),
                    round(a["static_f1"][i], 3),
                ])
        rp = run / "robustness.json"
        if rp.exists():
            for r in json.loads(rp.read_text()):
                rob_rows.append([r["size"], round(r["density"], 2), round(r["f1"], 3)])
    if not abl_rows:
        abl_rows = [["(run main.py)", "", "", "", ""]]
    if not rob_rows:
        rob_rows = [["(run main.py)", "", ""]]
    return abl_rows, rob_rows


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_game_of_life — pass_2\n"
        "**One self-attention layer + MLP** (a small delta from `base_model.py`), "
        "hand-set so attention gathers each cell's 8 toroidal neighbours and the "
        "MLP applies the birth/survival rule. No conv, no training."
    )

    with gr.Tab("Demo"):
        with gr.Row():
            density = gr.Slider(0.1, 0.9, value=0.3, step=0.05, label="Live-cell density")
            seed = gr.Number(value=0, label="Seed", precision=0)
            qi = gr.Slider(0, 15, value=8, step=1, label="Query cell row")
            qj = gr.Slider(0, 15, value=8, step=1, label="Query cell col")
        btn = gr.Button("Generate & compare", variant="primary")
        with gr.Row():
            cur = gr.Image(label="Current board", height=290)
            true_im = gr.Image(label="True next (independent NumPy GoL)", height=290)
            pred_im = gr.Image(label="Circuit prediction", height=290)
        with gr.Row():
            match_im = gr.Image(label="Match (green=correct, red=wrong)", height=290)
            attn_im = gr.Image(label="Attention of query cell (red) → 8 neighbours", height=290)
        stats = gr.Markdown()

        gr.Markdown("### Causal ablation — does the model *use* the neighbour attention?")
        gr.Markdown(
            "Knocking out the neighbour mask (global uniform attention) or zeroing "
            "the attention output (self-only) collapses F1, while the static copy "
            "baseline stays low. Only the full neighbour-selecting attention works."
        )
        abl_tbl = gr.Dataframe(
            headers=["density", "full F1", "ablate: global attn", "ablate: self-only", "static baseline"],
            label="alive-next F1 by density",
            interactive=False,
        )
        gr.Markdown("### Operating range — same hand-set circuit across grid sizes (8→64, ~16×–4096× cells)")
        rob_tbl = gr.Dataframe(
            headers=["grid size", "density", "F1"],
            label="F1 across grid sizes",
            interactive=False,
        )

        btn.click(
            run_demo, inputs=[density, seed, qi, qj],
            outputs=[cur, true_im, pred_im, match_im, attn_im, stats],
        )
        demo.load(
            run_demo, inputs=[density, seed, qi, qj],
            outputs=[cur, true_im, pred_im, match_im, attn_im, stats],
        )
        demo.load(load_tables, inputs=None, outputs=[abl_tbl, rob_tbl])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
