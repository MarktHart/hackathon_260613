"""Gradio app for attention_block_2d / pass_4.

Demo tab: pick one of the 16 canonical matrices and see (a) the attention
heatmap, (b) the *displacement footprint* — the key-minus-query offset map
that the geometric classifier actually reads — and (c) the prediction the
method recorded for it. The footprint is the crux: a tight 3x3 blob = local,
a spaced 3x3 blob = dilated, a cross = global, a half-plane = causal.

Benchmark tab: the shared cross-attempt leaderboard.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel, load_task

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent

task = load_task(__file__)
_batch = task.generate(0)
_MATS = _batch.matrices            # (16, 64, 64)
_PIDS = _batch.pattern_ids
_PARAMS = _batch.params

H, W = 8, 8
N = H * W
TAU = 0.4 / N

_CHOICES = [
    (f"#{i:02d}  {_PIDS[i]}  {_PARAMS[i]}", i) for i in range(len(_PIDS))
]


def _latest_benchmark():
    res = ATTEMPT_DIR / "results"
    if not res.is_dir():
        return None
    paths = sorted(res.glob("*/benchmark.json"))
    if not paths:
        return None
    return json.loads(paths[-1].read_text())


def _footprint(A: np.ndarray) -> np.ndarray:
    """Accumulate attention mass by (dr, dc) displacement onto a 15x15 grid."""
    rows = np.arange(N) // W
    cols = np.arange(N) % W
    M = A > TAU
    ii, jj = np.where(M)
    dr = rows[jj] - rows[ii] + 7
    dc = cols[jj] - cols[ii] + 7
    foot = np.zeros((15, 15), dtype=np.float64)
    np.add.at(foot, (dr, dc), A[ii, jj])
    return foot


def render(idx: int):
    idx = int(idx)
    A = _MATS[idx]
    foot = _footprint(A)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(A, cmap="magma", aspect="equal")
    axes[0].set_title("Attention matrix  A[query, key]")
    axes[0].set_xlabel("key index"); axes[0].set_ylabel("query index")

    im = axes[1].imshow(foot, cmap="viridis", extent=[-7.5, 7.5, 7.5, -7.5])
    axes[1].set_title("Displacement footprint  (dc, dr)")
    axes[1].set_xlabel("dc = col(key) - col(query)")
    axes[1].set_ylabel("dr = row(key) - row(query)")
    axes[1].axhline(0, color="w", lw=0.4); axes[1].axvline(0, color="w", lw=0.4)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()

    bench = _latest_benchmark()
    gt = f"**Ground truth:** `{_PIDS[idx]}`  params `{_PARAMS[idx]}`"
    if bench is None:
        info = gt + "\n\n_No benchmark.json yet — run `main.py` to populate predictions._"
    else:
        rec = bench["payload"]["sweep"][idx]
        ok = "✅ correct" if rec["correct"] else "❌ wrong"
        info = (
            gt
            + f"\n\n**Predicted:** `{rec['pred_pattern_id']}`  params "
            f"`{rec['pred_params']}`  →  {ok}"
            + f"\n\n**Confidence:** {rec['confidence']:.3f}"
        )
    return fig, info


def acc_bar():
    bench = _latest_benchmark()
    fams = ["local", "dilated", "global", "causal_2d"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if bench is None:
        ax.text(0.5, 0.5, "run main.py first", ha="center", va="center")
        ax.axis("off")
        return fig
    m = bench["metrics"]
    vals = [m.get(f"pattern_acc_{f}", 0.0) for f in fams]
    overall = m.get("pattern_acc_canonical", 0.0)
    base = m.get("linear_baseline_acc", 0.0)
    bars = ax.bar(fams, vals, color="#4c78a8")
    ax.axhline(base, color="crimson", ls="--", lw=1.4,
               label=f"majority baseline ({base:.2f})")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.set_title(f"Per-family accuracy  (overall {overall:.2f})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


with gr.Blocks(title="attention_block_2d / pass_4") as demo:
    gr.Markdown(
        "# attention_block_2d — geometric pattern recovery\n"
        "A hand-built circuit reads the **spatial structure** off the attention "
        "matrix: displacement offsets → window size & dilation, a full row+column "
        "→ a global token, a triangular mask → 2D causal. No ground-truth "
        "generators are consulted."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            dd = gr.Dropdown(choices=_CHOICES, value=0, label="Canonical example")
        with gr.Row():
            plot = gr.Plot(label="matrix + displacement footprint")
        info = gr.Markdown()
        gr.Markdown("### How the method scores across families")
        bar = gr.Plot(label="per-family accuracy")

        dd.change(render, inputs=dd, outputs=[plot, info])
        demo.load(render, inputs=dd, outputs=[plot, info])
        demo.load(acc_bar, inputs=None, outputs=bar)

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
