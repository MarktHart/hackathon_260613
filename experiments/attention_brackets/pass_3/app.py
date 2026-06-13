"""Gradio app for attention_brackets / pass_3.

Demo tab: (1) the stack-match head's attention heatmap for one sequence, with
the parser's true matching opener overlaid, and (2) a depth-sweep bar chart
comparing the stack-match head against the cheap heuristics it is meant to
beat (nearest-opener, uniform/random). The contrast is the whole point: the
stack head holds near-perfect accuracy as nesting deepens, while nearest-opener
collapses on the outer brackets it can't reach.

Benchmark tab: the shared cross-attempt panel.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel, load_task

task = load_task(__file__)

DEVICE = "cuda"
OPEN, CLOSE, PAD = 0, 1, 2
C = 300.0
ALPHA = 4.0

batch = task.generate(0)  # pure NumPy; safe at import / boot-check


# ---------------------------------------------------------------------------
# Model functions (duplicated from main.py so the app needs no GPU at import).
# ---------------------------------------------------------------------------
def _levels(tokens):
    L = len(tokens)
    level = np.zeros(L, dtype=np.int64)
    is_open = np.zeros(L, dtype=bool)
    is_close = np.zeros(L, dtype=bool)
    h = 0
    for i, t in enumerate(tokens):
        if t == OPEN:
            h += 1
            level[i] = h
            is_open[i] = True
        elif t == CLOSE:
            level[i] = h
            is_close[i] = True
            h = max(h - 1, 0)
    return level, is_open, is_close


def stack_match_model_fn(tokens):
    tokens = np.asarray(tokens).astype(np.int64)
    L = int(tokens.shape[0])
    level, is_open, is_close = _levels(tokens)
    D = L + 2
    lvl = torch.as_tensor(level, device=DEVICE)
    onehot = torch.zeros((L, D), device=DEVICE, dtype=torch.float32)
    onehot[torch.arange(L, device=DEVICE), lvl] = 1.0
    opent = torch.as_tensor(is_open, device=DEVICE, dtype=torch.float32)
    closet = torch.as_tensor(is_close, device=DEVICE, dtype=torch.float32)
    Q = onehot * closet[:, None]
    K = onehot * opent[:, None]
    same_level = Q @ K.t()
    pos = torch.arange(L, device=DEVICE, dtype=torch.float32)
    logits = C * same_level + ALPHA * pos[None, :]
    neg = torch.full_like(logits, -1e9)
    logits = torch.where(opent[None, :] > 0, logits, neg)
    causal = torch.tril(torch.ones((L, L), device=DEVICE))
    logits = torch.where(causal > 0, logits, neg)
    attn = torch.softmax(logits, dim=1)
    return attn.detach().cpu().numpy()


def nearest_opener_model_fn(tokens):
    """Strawman: every closer attends to the *nearest preceding* opener.

    Correct at depth 1, but wrong for outer brackets once nesting > 1."""
    tokens = np.asarray(tokens).astype(np.int64)
    L = int(tokens.shape[0])
    A = np.zeros((L, L), dtype=np.float32)
    last_open = -1
    for i, t in enumerate(tokens):
        if t == OPEN:
            last_open = i
        elif t == CLOSE and last_open >= 0:
            A[i, last_open] = 1.0
    At = torch.as_tensor(A, device=DEVICE)
    return (At * 1.0).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Figures.
# ---------------------------------------------------------------------------
def _seq_string(tokens):
    sym = {OPEN: "(", CLOSE: ")", PAD: "."}
    return "".join(sym.get(int(t), "?") for t in tokens)


def make_heatmap(depth, idx):
    depth = int(depth)
    idx = int(idx)
    tokens = np.asarray(batch.sequences[depth][idx], dtype=np.int64)
    match = np.asarray(batch.matches[depth][idx], dtype=np.int64)
    attn = stack_match_model_fn(tokens)
    L = len(tokens)

    closers = [i for i in range(L) if match[i] >= 0]
    hits = sum(int(np.argmax(attn[i]) == match[i]) for i in closers)
    acc = hits / max(1, len(closers))

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(attn, cmap="magma", vmin=0.0, vmax=1.0, aspect="equal")
    # Overlay the parser ground-truth: for each closer, ring its true opener.
    ys = [i for i in closers]
    xs = [int(match[i]) for i in closers]
    ax.scatter(xs, ys, s=70, facecolors="none", edgecolors="cyan",
               linewidths=1.6, label="true matching opener")
    ax.set_xlabel("key position (opener candidates)")
    ax.set_ylabel("query position (closing brackets)")
    ax.set_title(f"Stack-match attention — depth {depth}, seq {idx}\n"
                 f"argmax-on-true-opener accuracy = {acc:.0%}")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()
    return fig, _seq_string(tokens)


def _sweep_metrics(model_fn):
    payload = task.evaluate(model_fn)
    accs, lifts = [], []
    for rec in payload["sweep"]:
        base = rec["uniform_baseline_mass"]
        lift = max(0.0, (rec["match_mass"] - base) / max(1e-9, 1.0 - base))
        accs.append(rec["match_accuracy"])
        lifts.append(lift)
    return accs, lifts


def make_sweep_fig():
    depths = list(batch.depths)
    methods = [
        ("stack-match (ours)", stack_match_model_fn, "#1b9e77"),
        ("nearest-opener", nearest_opener_model_fn, "#d95f02"),
        ("random", task.random_model_fn(), "#7570b3"),
    ]
    x = np.arange(len(depths))
    w = 0.26

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for j, (name, fn, color) in enumerate(methods):
        accs, lifts = _sweep_metrics(fn)
        ax1.bar(x + (j - 1) * w, accs, w, label=name, color=color)
        ax2.bar(x + (j - 1) * w, lifts, w, label=name, color=color)

    for ax, title, ylab in (
        (ax1, "argmax matching accuracy", "accuracy"),
        (ax2, "normalised lift over uniform", "lift  (1 = all mass on match)"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([f"d{d}" for d in depths])
        ax.set_xlabel("nesting depth")
        ax.set_ylabel(ylab)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    ax1.legend(fontsize=8, loc="lower left")
    fig.suptitle("Stack-matching survives depth; the cheap heuristics do not")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Blocks.
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_brackets · pass_3\n"
        "A hand-built **stack-matching attention head**: each closing bracket's "
        "query routes to the opener a parser would pop, via one-hot *nesting-level* "
        "Q·K plus a recency tiebreak — no oracle, no training."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                depth_dd = gr.Dropdown(
                    choices=[int(d) for d in batch.depths], value=3,
                    label="Nesting depth",
                )
                idx_sl = gr.Slider(
                    0, int(batch.n_per_depth) - 1, value=0, step=1,
                    label="Sequence index",
                )
            seqbox = gr.Textbox(label="bracket sequence", interactive=False)
            heat = gr.Plot(label="Stack-match attention (cyan rings = true opener)")
            gr.Markdown(
                "### Across the depth sweep\n"
                "The stack head stays near-perfect as nesting grows; the "
                "**nearest-opener** heuristic is right at depth 1 but collapses on "
                "the outer brackets it can't reach, and **random** sits at the floor."
            )
            sweep_plot = gr.Plot(label="Accuracy & lift by depth")

        with gr.TabItem("Benchmark"):
            benchmark_panel(Path(__file__).parent.parent)

    def _update(depth, idx):
        fig, s = make_heatmap(depth, idx)
        return fig, s

    depth_dd.change(_update, [depth_dd, idx_sl], [heat, seqbox])
    idx_sl.change(_update, [depth_dd, idx_sl], [heat, seqbox])
    demo.load(_update, [depth_dd, idx_sl], [heat, seqbox])
    demo.load(make_sweep_fig, None, sweep_plot)


if __name__ == "__main__":
    demo.launch()
