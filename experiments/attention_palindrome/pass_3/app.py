"""Gradio app for attention_palindrome / pass_3.

Demo tab  : makes the hand-built mirror mechanism legible — the anti-diagonal
            attention routing, and an AUC-vs-difficulty bar chart that contrasts
            the mirror head against the histogram baseline and ablated routings.
Benchmark : shared cross-attempt leaderboard / history panel.
"""

from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import gradio as gr

from agentic.experiments import load_task, benchmark_panel

task = load_task(__file__)
ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)

# App may run where no GPU is reserved (the boot-check just imports the module),
# so here — unlike main.py — fall back to CPU gracefully.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEQ_LEN = int(task.SEQ_LEN)
VOCAB = int(task.VOCAB)
TEMP = 30.0
SWEEP = list(task.MISMATCH_SWEEP)


# ---------------------------------------------------------------------------
# The hand-built circuit (same maths as main.py) — runs live for the demo.
# ---------------------------------------------------------------------------
def attention_pattern(routing: str = "mirror", temp: float = TEMP) -> torch.Tensor:
    L = SEQ_LEN
    idx = torch.arange(L, device=DEVICE)
    if routing == "mirror":
        target = (L - 1) - idx
    elif routing == "identity":
        target = idx
    elif routing == "shift":
        target = ((L - 2) - idx) % L
    else:
        raise ValueError(routing)
    K_pos = torch.eye(L, device=DEVICE)
    Q_pos = torch.eye(L, device=DEVICE)[target]
    return torch.softmax((Q_pos @ K_pos.t()) * temp, dim=-1)


def circuit_scores(tokens_np: np.ndarray, routing: str = "mirror") -> np.ndarray:
    tokens = torch.as_tensor(tokens_np, dtype=torch.long, device=DEVICE)
    E = torch.nn.functional.one_hot(tokens, num_classes=VOCAB).float()
    A = attention_pattern(routing)
    O = torch.einsum("ij,bjv->biv", A, E)
    match = (E * O).sum(dim=-1)
    return match.sum(dim=-1).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------
def list_runs():
    runs = sorted(glob.glob(os.path.join(ATTEMPT_DIR, "results", "*")))
    return [os.path.basename(r) for r in runs if os.path.isdir(r)]


def latest_run():
    runs = list_runs()
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_attention(routing: str = "mirror"):
    A = attention_pattern(routing).detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(A, cmap="magma", vmin=0, vmax=1)
    ax.set_title(f"Attention routing ({routing}): query i → key", fontsize=11)
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_xticks(range(0, SEQ_LEN, 2))
    ax.set_yticks(range(0, SEQ_LEN, 2))
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    fig.tight_layout()
    return fig


def _auc_by_k(scores, batch):
    recs = task._sweep_records(scores, batch)
    return {int(r["mismatch"]): float(r["auc"]) for r in recs}


def plot_auc_bars(seed: int):
    seed = int(seed)
    batch = task.generate(seed)
    model = _auc_by_k(circuit_scores(batch.tokens, "mirror"), batch)
    base = _auc_by_k(task._ridge_baseline_scores(batch), batch)
    ablate = _auc_by_k(circuit_scores(batch.tokens, "identity"), batch)

    ks = SWEEP
    x = np.arange(len(ks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.bar(x - w, [model[k] for k in ks], w, label="mirror head (ours)", color="#1b9e77")
    ax.bar(x, [base[k] for k in ks], w, label="histogram baseline", color="#7570b3")
    ax.bar(x + w, [ablate[k] for k in ks], w, label="ablation: identity routing", color="#d95f02")
    ax.axhline(0.5, ls="--", lw=1, color="grey")
    ax.text(len(ks) - 1.4, 0.51, "chance", color="grey", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("rank-AUC  (positives vs k-broken negatives)")
    ax.set_xlabel("broken mirror pairs  k  (k=1 = hardest / most diagnostic)")
    ax.set_title(f"Palindrome separation by difficulty  (seed={seed})", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def example_text(seed: int):
    seed = int(seed)
    batch = task.generate(seed)
    scores = circuit_scores(batch.tokens, "mirror")
    pos_i = int(np.flatnonzero(batch.mismatch == 0)[0])
    neg_i = int(np.flatnonzero(batch.mismatch == 1)[0])

    def render(i, label):
        toks = batch.tokens[i].tolist()
        L = SEQ_LEN
        pairs = []
        for p in range(L // 2):
            a, b = toks[p], toks[L - 1 - p]
            mark = "=" if a == b else "≠"
            pairs.append(f"({a}{mark}{b})")
        return (f"[{label}]  score={scores[i]:.1f} / {SEQ_LEN}\n"
                f"  tokens : {toks}\n"
                f"  pairs  : {' '.join(pairs)}")

    return (render(pos_i, "perfect palindrome") + "\n\n"
            + render(neg_i, "k=1 near-palindrome (one pair broken)") + "\n\n"
            f"The mirror head scores each sequence by how many of the {SEQ_LEN} "
            f"positions agree with their mirror. A single broken pair drops the "
            f"score by exactly 2 — enough to rank every positive above every "
            f"k=1 negative (AUC = 1.0), where the token histogram is at chance.")


def refresh(seed):
    return plot_attention("mirror"), plot_auc_bars(seed), example_text(seed)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="attention_palindrome / pass_3") as demo:
    gr.Markdown(
        "# Palindrome detection by a hand-built mirror-comparison head\n"
        "A single attention head (zero training) whose Q/K depend only on "
        "**position**, wired so query `i` attends to key `L-1-i`. The value "
        "carries token identity, so the head reads the *mirror* token and a dot "
        "product checks equality. The palindrome score is the number of agreeing "
        "positions. This separates perfect palindromes from near-palindromes "
        "**even when only one mirror pair is broken** — exactly where a "
        "bag-of-tokens readout collapses to chance."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                seed_in = gr.Number(value=42, precision=0, label="Seed (data batch)")
                go = gr.Button("Recompute", variant="primary")
            with gr.Row():
                attn_plot = gr.Plot(label="The mechanism: anti-diagonal routing")
                auc_plot = gr.Plot(label="Claim: AUC stays at 1.0 across difficulty")
            ex_box = gr.Textbox(label="Worked example", lines=9)
            gr.Markdown(
                "**How to read this.** *Left:* the attention weights form a clean "
                "anti-diagonal — position `i` routes all its mass to `L-1-i`. "
                "*Right:* the green mirror head sits at AUC 1.0 for every `k`, "
                "including the diagnostic `k=1`; the purple histogram baseline and "
                "the orange *identity-routing ablation* (head attends to itself "
                "instead of the mirror) both sit at chance. The ablation is the "
                "causal check: delete the mirror routing and the mechanism dies."
            )

            go.click(refresh, inputs=[seed_in], outputs=[attn_plot, auc_plot, ex_box])
            demo.load(refresh, inputs=[seed_in], outputs=[attn_plot, auc_plot, ex_box])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
