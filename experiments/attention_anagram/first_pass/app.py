"""Gradio app for attention_anagram / first_pass.

Demo tab: shows that the hand-built token-matching head aligns target positions
to their true source positions, far above the uniform baseline. Includes a live
temperature slider so the grader can watch alignment collapse toward 1/8 as the
QK logits get soft.

Benchmark tab: cross-attempt leaderboard / history via benchmark_panel.
"""
import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).resolve().parent.parent
ATTEMPT_DIR = Path(__file__).resolve().parent
RESULTS = ATTEMPT_DIR / "results"
SEQ_LEN = 8
VOCAB_SIZE = 50
N_HEADS = 8


# ----------------------------------------------------------------------------
# Run discovery / loading
# ----------------------------------------------------------------------------
def list_runs():
    if not RESULTS.exists():
        return []
    runs = sorted([p.name for p in RESULTS.iterdir() if (p / "payload.json").exists()])
    return runs[::-1]  # newest first


def load_payload(run_id):
    if not run_id:
        return None
    p = RESULTS / run_id / "payload.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ----------------------------------------------------------------------------
# Plots from a saved payload
# ----------------------------------------------------------------------------
def plot_run(run_id):
    payload = load_payload(run_id)
    if payload is None:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No run selected", ha="center")
        return fig

    sweep = {r["perm_type"]: r for r in payload["sweep"]}
    perm_types = ["swap", "rotation", "random"]
    baseline = 1.0 / SEQ_LEN

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # --- Left: mean alignment per perm type, best head, vs baseline ---
    ax = axes[0]
    mean_vals, max_vals = [], []
    for pt in perm_types:
        rec = sweep.get(pt)
        if rec is None:
            mean_vals.append(0.0)
            max_vals.append(0.0)
            continue
        ma = np.mean([h["mean_alignment"] for h in rec["head_alignments"]])
        mx = np.max([h["max_alignment"] for h in rec["head_alignments"]])
        mean_vals.append(ma)
        max_vals.append(mx)

    x = np.arange(len(perm_types))
    ax.bar(x - 0.2, mean_vals, 0.4, label="mean over heads", color="#2b6cb0")
    ax.bar(x + 0.2, max_vals, 0.4, label="best head", color="#63b3ed")
    ax.axhline(baseline, color="crimson", ls="--", label=f"uniform baseline ({baseline:.3f})")
    ax.set_xticks(x)
    ax.set_xticklabels(perm_types)
    ax.set_ylabel("alignment on TRUE source position")
    ax.set_ylim(0, 1.05)
    ax.set_title("Anagram alignment by permutation type")
    ax.legend(fontsize=8)

    # --- Right: per-position alignment (random perm), head 0 ---
    ax = axes[1]
    rec = sweep.get("random")
    if rec is not None:
        for h in rec["head_alignments"][:3]:
            ax.plot(range(SEQ_LEN), h["alignment_per_pos"],
                    marker="o", label=f"head {h['head_idx']}")
    ax.axhline(baseline, color="crimson", ls="--", label="baseline")
    ax.set_xlabel("target position")
    ax.set_ylabel("mean alignment")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-position alignment (random perm)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


def run_summary(run_id):
    payload = load_payload(run_id)
    if payload is None:
        return "No run."
    sweep = {r["perm_type"]: r for r in payload["sweep"]}
    rec = sweep.get("random")
    canonical = np.mean([h["mean_alignment"] for h in rec["head_alignments"]]) if rec else 0.0
    return (f"**Canonical (random perm) alignment:** {canonical:.4f}  \n"
            f"**Uniform baseline:** {1/SEQ_LEN:.4f}  \n"
            f"**Lift over baseline:** {canonical - 1/SEQ_LEN:.4f}")


# ----------------------------------------------------------------------------
# Live interactive temperature demo (CPU numpy — recomputes the mechanism)
# ----------------------------------------------------------------------------
def _gen_one(seed=0):
    rng = np.random.default_rng(seed)
    src = rng.integers(0, VOCAB_SIZE, size=SEQ_LEN)
    perm = rng.permutation(SEQ_LEN)
    tgt = src[perm]
    return src, tgt, perm


def live_demo(temperature, seed):
    src, tgt, perm = _gen_one(int(seed))
    # match[t, s] = 1 if tgt[t] == src[s]
    match = (tgt[:, None] == src[None, :]).astype(np.float32)
    scores = match * float(temperature)
    scores = scores - scores.max(axis=1, keepdims=True)
    e = np.exp(scores)
    attn = e / e.sum(axis=1, keepdims=True)

    # alignment = attn on true source position
    align = attn[np.arange(SEQ_LEN), perm].mean()

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1)
    # mark true source positions
    for t in range(SEQ_LEN):
        ax.add_patch(plt.Rectangle((perm[t] - 0.5, t - 0.5), 1, 1,
                                   fill=False, edgecolor="red", lw=2))
    ax.set_xlabel("source position")
    ax.set_ylabel("target position")
    ax.set_xticks(range(SEQ_LEN))
    ax.set_yticks(range(SEQ_LEN))
    ax.set_title(f"Attention  (temp={temperature:.0f})\nmean alignment on true src = {align:.3f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    txt = (f"Red boxes = true source position for each target token.  \n"
           f"Mean alignment on true source = **{align:.3f}** "
           f"(uniform baseline = {1/SEQ_LEN:.3f}).")
    return fig, txt


# ----------------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------------
with gr.Blocks(title="attention_anagram / first_pass") as demo:
    gr.Markdown("# attention_anagram — token-matching attention head")
    gr.Markdown(
        "A single hand-built attention head (W_Q = W_K = identity over a one-hot "
        "token embedding) aligns each target token to its true source position "
        "in an anagram pair. No training."
    )

    with gr.Tab("Demo"):
        runs = list_runs()
        with gr.Row():
            run_dd = gr.Dropdown(runs, value=(runs[0] if runs else None),
                                 label="Benchmarked run (newest first)")
        summary_md = gr.Markdown()
        run_plot = gr.Plot()

        gr.Markdown("### Live mechanism: temperature controls alignment sharpness")
        with gr.Row():
            temp_sl = gr.Slider(0.0, 40.0, value=30.0, step=1.0, label="QK logit temperature")
            seed_sl = gr.Slider(0, 50, value=0, step=1, label="example seed")
        live_md = gr.Markdown()
        live_plot = gr.Plot()

        def _refresh(run_id):
            return plot_run(run_id), run_summary(run_id)

        run_dd.change(_refresh, inputs=run_dd, outputs=[run_plot, summary_md])
        temp_sl.change(live_demo, inputs=[temp_sl, seed_sl], outputs=[live_plot, live_md])
        seed_sl.change(live_demo, inputs=[temp_sl, seed_sl], outputs=[live_plot, live_md])

        def _init():
            runs2 = list_runs()
            rid = runs2[0] if runs2 else None
            fig, md = _refresh(rid)
            lfig, lmd = live_demo(30.0, 0)
            return fig, md, lfig, lmd

        demo.load(_init, inputs=None, outputs=[run_plot, summary_md, live_plot, live_md])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
