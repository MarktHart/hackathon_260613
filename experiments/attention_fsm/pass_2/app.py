"""attention_fsm / pass_2 — Gradio app: Demo tab + Benchmark tab.

The Demo makes one claim legible: a single causal attention head (prefix-sum of
group increments) *is* the DFA tracker. Knock the head out and accuracy falls to
chance; the s0 anchor alone tracks nothing.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"
TOKEN_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
CHANCE = 1.0 / 3.0


def list_runs():
    if not RESULTS.exists():
        return []
    runs = [d.name for d in RESULTS.iterdir() if (d / "artifacts.json").exists()]
    return sorted(runs, reverse=True)


def load_run(run_name):
    art = json.loads((RESULTS / run_name / "artifacts.json").read_text())
    tr = np.load(RESULTS / run_name / "traces.npz")
    return art, tr


def fig_bars(art):
    bars = art["bars"]
    labels = ["Full head\n(prefix-sum)", "Head ablated\n(s0 only)", "Random\nbaseline"]
    vals = [bars["full"], bars["ablated_head"], bars["random"]]
    colors = ["#2a9d8f", "#e76f51", "#999999"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(labels, vals, color=colors)
    ax.axhline(CHANCE, ls="--", c="k", lw=1, label="chance (1/3)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("post-burn-in accuracy")
    ax.set_title("Ablating the head collapses tracking to chance")
    ax.legend(loc="center right")
    fig.tight_layout()
    return fig


def fig_perpos(art):
    pp = art["per_position"]
    bn = art["burnin"]
    x = np.arange(len(pp["full"]))
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(x, pp["full"], color="#2a9d8f", lw=2, label="full head")
    ax.plot(x, pp["ablated_head"], color="#e76f51", lw=2, label="head ablated")
    ax.axhline(CHANCE, ls="--", c="k", lw=1, label="chance")
    ax.axvspan(0, bn, color="grey", alpha=0.12, label=f"burn-in (<{bn})")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("position in sequence")
    ax.set_ylabel("accuracy")
    ax.set_title("Per-position accuracy holds flat with depth")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_oprange(art):
    o = art["operating_range"]
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(o["lengths"], o["full"], "o-", color="#2a9d8f", label="full head")
    ax.plot(o["lengths"], o["ablated_head"], "o-", color="#e76f51", label="head ablated")
    ax.axhline(CHANCE, ls="--", c="k", lw=1, label="chance")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("sequence length (log scale)")
    ax.set_ylabel("accuracy")
    ax.set_title("Operating range: exact from L=8 to L=1024")
    ax.legend(loc="center right", fontsize=8)
    fig.tight_layout()
    return fig


def fig_pattern(seq_len=20):
    pat = np.tril(np.ones((seq_len, seq_len)))
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.imshow(pat, cmap="Greens", aspect="equal")
    ax.set_title("Attention pattern = tril(ones)\n(value = token increment)")
    ax.set_xlabel("key position i")
    ax.set_ylabel("query position t")
    fig.tight_layout()
    return fig


def trace_md(tr, seq_idx, max_pos=28):
    tok = tr["tokens"][seq_idx]
    true = tr["true_states"][seq_idx]
    pf = tr["preds_full"][seq_idx]
    pa = tr["preds_ablated"][seq_idx]
    s0 = int(tr["s0"][seq_idx])
    inc = [0, 1, 2, 1]
    n = min(max_pos, len(tok))
    rows = [
        f"**Sequence {seq_idx}** — start state s0 = **{s0}**  "
        f"(supplied boundary condition; everything else computed from tokens)\n",
        "| pos | token | increment | true state | full pred | ablated pred |",
        "|----:|:-----:|:---------:|:----------:|:---------:|:------------:|",
    ]
    for t in range(n):
        i = 0 if t == 0 else inc[int(tok[t])]
        fmark = "✓" if pf[t] == true[t] else "✗"
        amark = "✓" if pa[t] == true[t] else "✗"
        rows.append(
            f"| {t} | {TOKEN_LETTER[int(tok[t])]} | +{i} | {int(true[t])} | "
            f"{int(pf[t])} {fmark} | {int(pa[t])} {amark} |"
        )
    return "\n".join(rows)


def update(run_name, seq_idx):
    if not run_name:
        empty = plt.figure()
        return empty, empty, empty, fig_pattern(), "No runs found — run main.py first."
    art, tr = load_run(run_name)
    seq_idx = int(min(max(0, seq_idx), tr["tokens"].shape[0] - 1))
    return (
        fig_bars(art),
        fig_perpos(art),
        fig_oprange(art),
        fig_pattern(),
        trace_md(tr, seq_idx),
    )


with gr.Blocks(title="attention_fsm / pass_2") as demo:
    gr.Markdown(
        "# attention_fsm — pass_2: a single attention head tracks the DFA\n"
        "The DFA is a **Z/3 permutation automaton**: `state[t] = (s0 + Σ inc(token_i)) mod 3`, "
        "`inc=[0,1,2,1]`. One causal head with pattern `tril(ones)` and value = token increment "
        "computes the running sum exactly. Because every token is a bijection, the start state "
        "`s0` is never revealed by tokens, so it is supplied as a boundary condition — but the "
        "**tracking** is done by the head. Ablate it and accuracy falls to chance."
    )

    runs = list_runs()
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(runs, value=(runs[0] if runs else None), label="Run")
            seq_sl = gr.Slider(0, 127, value=0, step=1, label="Sequence index (for trace)")
        with gr.Row():
            bars = gr.Plot(label="Headline: full vs ablated vs random")
            perpos = gr.Plot(label="Per-position accuracy")
        with gr.Row():
            oprange = gr.Plot(label="Operating range (length)")
            pattern = gr.Plot(label="Attention pattern")
        trace = gr.Markdown()

        run_dd.change(update, [run_dd, seq_sl], [bars, perpos, oprange, pattern, trace])
        seq_sl.change(update, [run_dd, seq_sl], [bars, perpos, oprange, pattern, trace])
        demo.load(update, [run_dd, seq_sl], [bars, perpos, oprange, pattern, trace])

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
