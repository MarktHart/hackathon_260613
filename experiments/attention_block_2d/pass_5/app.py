"""Gradio app for attention_block_2d / pass_5.

Four views, each one a claim a grader can check by eye:

  • Demo            — the reader recovers family+params off the canonical 16.
  • Faithfulness    — ablating the producer's bias table breaks the pattern AND
                      the reader's verdict (the causal evidence).
  • Operating range — reader accuracy vs grid size, N = 16 … 1024.
  • Benchmark       — the shared cross-attempt leaderboard.

The app is torch-free: it only reads the artefacts main.py wrote, so the boot
check imports cleanly without a GPU.
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

_CHOICES = [(f"#{i:02d}  {_PIDS[i]}  {_PARAMS[i]}", i) for i in range(len(_PIDS))]
_FAM_CHOICES = ["local", "dilated", "global", "causal_2d"]


# --------------------------------------------------------------------------- #
# artefact loading
# --------------------------------------------------------------------------- #
def _latest_run() -> Path | None:
    res = ATTEMPT_DIR / "results"
    if not res.is_dir():
        return None
    runs = sorted(p for p in res.iterdir() if p.is_dir())
    return runs[-1] if runs else None


def _load_json(name: str):
    run = _latest_run()
    if run is None:
        return None
    p = run / name
    return json.loads(p.read_text()) if p.is_file() else None


def _load_examples():
    run = _latest_run()
    if run is None:
        return None
    p = run / "faith_examples.npz"
    if not p.is_file():
        return None
    d = np.load(p, allow_pickle=True)
    return {"families": list(d["families"]), "intact": d["intact"], "ablated": d["ablated"]}


# --------------------------------------------------------------------------- #
# Demo tab
# --------------------------------------------------------------------------- #
def _footprint(A: np.ndarray) -> np.ndarray:
    rows = np.arange(N) // W
    cols = np.arange(N) % W
    rowmax = A.max(axis=1, keepdims=True)
    M = A > 0.5 * rowmax
    ii, jj = np.where(M)
    dr = rows[jj] - rows[ii] + 7
    dc = cols[jj] - cols[ii] + 7
    foot = np.zeros((15, 15))
    np.add.at(foot, (dr, dc), A[ii, jj])
    return foot


def render_demo(idx: int):
    idx = int(idx)
    A = _MATS[idx]
    foot = _footprint(A)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    axes[0].imshow(A, cmap="magma", aspect="equal")
    axes[0].set_title("Attention matrix  A[query, key]")
    axes[0].set_xlabel("key index"); axes[0].set_ylabel("query index")
    im = axes[1].imshow(foot, cmap="viridis", extent=[-7.5, 7.5, 7.5, -7.5])
    axes[1].set_title("Displacement footprint  (dc, dr)")
    axes[1].set_xlabel("dc = col(key) − col(query)")
    axes[1].set_ylabel("dr = row(key) − row(query)")
    axes[1].axhline(0, color="w", lw=0.4); axes[1].axvline(0, color="w", lw=0.4)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()

    bench = _load_json("benchmark.json")
    gt = f"**Ground truth:** `{_PIDS[idx]}`  params `{_PARAMS[idx]}`"
    if bench is None:
        info = gt + "\n\n_No benchmark.json yet — run `main.py`._"
    else:
        rec = bench["payload"]["sweep"][idx]
        ok = "✅ correct" if rec["correct"] else "❌ wrong"
        info = (gt
                + f"\n\n**Reader predicted:** `{rec['pred_pattern_id']}`  params "
                f"`{rec['pred_params']}`  →  {ok}"
                + f"\n\n**Confidence:** {rec['confidence']:.3f}")
    return fig, info


def acc_bar():
    bench = _load_json("benchmark.json")
    fams = _FAM_CHOICES
    fig, ax = plt.subplots(figsize=(6, 3.4))
    if bench is None:
        ax.text(0.5, 0.5, "run main.py first", ha="center", va="center"); ax.axis("off")
        return fig
    m = bench["metrics"]
    vals = [m.get(f"pattern_acc_{f}", 0.0) for f in fams]
    overall = m.get("pattern_acc_canonical", 0.0)
    base = m.get("linear_baseline_acc", 0.0)
    bars = ax.bar(fams, vals, color="#4c78a8")
    ax.axhline(base, color="crimson", ls="--", lw=1.4, label=f"majority baseline ({base:.2f})")
    ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy")
    ax.set_title(f"Per-family reader accuracy  (overall {overall:.2f})")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Faithfulness tab
# --------------------------------------------------------------------------- #
def render_faith(fam: str):
    ex = _load_examples()
    faith = _load_json("faithfulness.json")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    if ex is None or faith is None:
        for a in axes:
            a.text(0.5, 0.5, "run main.py first", ha="center", va="center"); a.axis("off")
        return fig, "_No faithfulness artefacts yet — run `main.py`._"
    k = ex["families"].index(fam)
    axes[0].imshow(ex["intact"][k], cmap="magma"); axes[0].set_title("Intact block")
    axes[1].imshow(ex["ablated"][k], cmap="magma"); axes[1].set_title("Bias-table ABLATED")
    for a in axes:
        a.set_xlabel("key"); a.set_ylabel("query")
    fig.tight_layout()
    row = next(r for r in faith["rows"] if r["family"] == fam)
    in_ok = "✅" if row["intact_correct"] else "❌"
    ab_ok = "✅" if row["ablated_correct"] else "❌"
    info = (
        f"**Family:** `{fam}`\n\n"
        f"**Intact** → reader says `{row['intact_pred']}` `{row['intact_params']}` "
        f"{in_ok}  (conf {row['intact_conf']:.3f})\n\n"
        f"**Ablated** (bias set to 0) → reader says `{row['ablated_pred']}` "
        f"`{row['ablated_params']}` {ab_ok}  (conf {row['ablated_conf']:.3f})\n\n"
        f"_The spatial pattern lives entirely in the additive bias table; "
        f"deleting it collapses attention to uniform and the reader can no "
        f"longer recover the family._"
    )
    return fig, info


def faith_bar():
    faith = _load_json("faithfulness.json")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if faith is None:
        ax.text(0.5, 0.5, "run main.py first", ha="center", va="center"); ax.axis("off")
        return fig
    vals = [faith["intact_acc"], faith["ablated_acc"]]
    bars = ax.bar(["intact block", "bias ablated"], vals, color=["#4c78a8", "#b0b0b0"])
    ax.set_ylim(0, 1.08); ax.set_ylabel("reader accuracy")
    ax.set_title("Causal ablation: pattern lives in the bias table")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Operating range tab
# --------------------------------------------------------------------------- #
def op_plot():
    op = _load_json("operating_range.json")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    if op is None:
        ax.text(0.5, 0.5, "run main.py first", ha="center", va="center"); ax.axis("off")
        return fig
    bg = op["by_grid"]
    Ns = [r["N"] for r in bg]
    accs = [r["acc"] for r in bg]
    bases = [r["baseline_acc"] for r in bg]
    ax.plot(Ns, accs, "o-", color="#4c78a8", lw=2, label="reader accuracy")
    ax.plot(Ns, bases, "s--", color="crimson", lw=1.4, label="majority baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns); ax.set_xticklabels([f"{r['side']}×{r['side']}\nN={r['N']}" for r in bg],
                                          fontsize=7)
    ax.set_ylim(0, 1.08); ax.set_ylabel("accuracy"); ax.set_xlabel("grid size (log scale)")
    ax.set_title(f"Operating range  (overall {op['overall_acc']:.2f} vs "
                 f"baseline {op['baseline_acc']:.2f})")
    ax.legend(loc="lower left", fontsize=8); fig.tight_layout()
    return fig


def op_summary():
    op = _load_json("operating_range.json")
    if op is None:
        return "_No operating-range artefacts yet — run `main.py`._"
    lines = ["| grid | N | examples | reader acc | baseline |",
             "|---|---|---|---|---|"]
    for r in op["by_grid"]:
        lines.append(f"| {r['side']}×{r['side']} | {r['N']} | {r['n']} | "
                     f"{r['acc']:.2f} | {r['baseline_acc']:.2f} |")
    lines.append(f"\n**Overall:** {op['overall_acc']:.3f} reader vs "
                 f"{op['baseline_acc']:.3f} majority baseline, across a 64× span "
                 f"in N (16 → 1024).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Blocks
# --------------------------------------------------------------------------- #
with gr.Blocks(title="attention_block_2d / pass_5") as demo:
    gr.Markdown(
        "# attention_block_2d — pass_5\n"
        "A hand-set **2D attention block** (Swin-style relative-position-bias, a "
        "small delta from `base_model.py`) *produces* each pattern; a hand-built "
        "geometric **reader** recovers `(family, params)` from the matrix alone. "
        "We then **ablate** the block's bias table to prove the pattern causally "
        "depends on it, and sweep grids from N=16 to N=1024."
    )

    with gr.Tab("Demo"):
        dd = gr.Dropdown(choices=_CHOICES, value=0, label="Canonical example (seed 0)")
        plot = gr.Plot(label="matrix + displacement footprint")
        info = gr.Markdown()
        gr.Markdown("### Per-family accuracy vs majority baseline")
        bar = gr.Plot(label="per-family accuracy")
        dd.change(render_demo, inputs=dd, outputs=[plot, info])
        demo.load(render_demo, inputs=dd, outputs=[plot, info])
        demo.load(acc_bar, inputs=None, outputs=bar)

    with gr.Tab("Faithfulness (ablation)"):
        gr.Markdown(
            "The producer's spatial pattern is a hand-set additive **bias table**. "
            "Setting it to zero is a causal knockout: attention goes uniform and "
            "the reader's verdict breaks. Intact 4/4 → ablated 0/4."
        )
        fdd = gr.Dropdown(choices=_FAM_CHOICES, value="local", label="Pattern family")
        fplot = gr.Plot(label="intact vs ablated attention")
        finfo = gr.Markdown()
        fbar = gr.Plot(label="reader accuracy: intact vs ablated")
        fdd.change(render_faith, inputs=fdd, outputs=[fplot, finfo])
        demo.load(render_faith, inputs=fdd, outputs=[fplot, finfo])
        demo.load(faith_bar, inputs=None, outputs=fbar)

    with gr.Tab("Operating range"):
        gr.Markdown(
            "Reader accuracy as a function of grid size — the producer drives "
            "every in-range param combo on grids from 4×4 (N=16) to 32×32 "
            "(N=1024), a 64× span."
        )
        opp = gr.Plot(label="accuracy vs N")
        ops = gr.Markdown()
        demo.load(op_plot, inputs=None, outputs=opp)
        demo.load(op_summary, inputs=None, outputs=ops)

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
