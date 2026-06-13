"""
Gradio app for attention_int_add / pass_3 — carry propagation AS ATTENTION.

Demo tab
  * Carry-lookahead attention probe: type a + b, see the (queries=answer columns,
    keys=source columns) attention heatmap. On a long carry chain like 999+1 the
    THOUSANDS query attends two columns down, PAST the 9-propagators, to the
    units column — the carry chain rendered as one attention pattern, no loop.
  * Faithfulness bars: full circuit vs carry-attention-ABLATED exact-match per
    carry slice (averaged over 8 held-out seeds). Removing the attention layer
    = the task's linear baseline.
  * Operating-range line: exact-match vs operand digit width (3..12), carry
    chains up to 4x the canonical length, vs the linear baseline.
Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""

import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS_DIR = ATTEMPT_DIR / "results"

POS_SCALE, MASK_BIG, PROP_BIG = 10.0, 1.0e4, 1.0e3


# ---- numpy mirror of the carry-lookahead attention (for the live probe) ----
def carry_lookahead_np(s):
    """s: (D,) column sums LSB-first. Returns (carry (D+1,), weights (D+1,D))."""
    D = len(s)
    g = (s >= 10).astype(float)
    p = (s == 9).astype(float)
    j = np.arange(D)
    weights = np.zeros((D + 1, D))
    carry = np.zeros(D + 1)
    for i in range(D + 1):
        score = POS_SCALE * j - MASK_BIG * (j >= i) - PROP_BIG * p
        score = score - score.max()
        w = np.exp(score)
        w = w / w.sum()
        weights[i] = w
        carry[i] = (w * g).sum()
    carry[0] = 0.0
    return np.round(carry), weights


def latest(name):
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(RESULTS_DIR.glob(f"*/{name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        return None
    with open(runs[0]) as f:
        return json.load(f)


# ---- interactive probe: attention heatmap + prediction ----
def probe(a, b):
    a, b = int(a), int(b)
    if not (0 <= a <= 999 and 0 <= b <= 999):
        fig, ax = plt.subplots(); ax.text(0.5, 0.5, "operands in [0,999]", ha="center"); ax.axis("off")
        return fig, "Enter operands in [0, 999]."
    da = [(a // 10 ** c) % 10 for c in range(3)]   # LSB-first
    db = [(b // 10 ** c) % 10 for c in range(3)]
    s = np.array([da[c] + db[c] for c in range(3)])
    carry, weights = carry_lookahead_np(s)
    digits = [(s[c] + int(carry[c])) % 10 for c in range(3)]
    lead = int(carry[3])
    pred_val = lead * 1000 + digits[2] * 100 + digits[1] * 10 + digits[0]
    true = a + b
    n_carry = int(sum(s[c] + carry[c] >= 10 for c in range(3)))

    # heatmap: rows = queries (carry into tens/hundreds/thousands), cols = keys
    qnames = ["units c-in", "tens c-in", "hundreds c-in", "THOUSANDS (lead)"]
    knames = ["units (c0)", "tens (c1)", "hundreds (c2)"]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    im = ax.imshow(weights, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(knames)
    ax.set_yticks(range(4)); ax.set_yticklabels(qnames)
    ax.set_xlabel("KEY: source column (reads generate signal g_j = [s_j ≥ 10])")
    ax.set_ylabel("QUERY: carry into column")
    ax.set_title(f"{a} + {b}: carry-lookahead attention\n"
                 "(propagator columns s=9 are skipped)")
    for i in range(4):
        for jx in range(3):
            v = weights[i, jx]
            ax.text(jx, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v < 0.5 else "black", fontsize=8)
    # annotate propagator columns
    for jx in range(3):
        if s[jx] == 9:
            ax.text(jx, -0.62, "PROP(9)", ha="center", color="#b00", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    fig.tight_layout()

    rows = ["### {} + {} = {}".format(a, b, true),
            f"**{n_carry} carry(ies)** in this problem.",
            "",
            "| column | a | b | s=a+b | carry-in (attn read) | digit |",
            "|---|---|---|---|---|---|"]
    cnames = ["units", "tens", "hundreds"]
    for c in range(3):
        rows.append(f"| {cnames[c]} | {da[c]} | {db[c]} | {s[c]} | "
                    f"{int(carry[c])} | {digits[c]} |")
    rows.append(f"| thousands | – | – | – | {lead} | {lead} |")
    rows += ["",
             f"**Prediction:** {pred_val}  {'✅' if pred_val == true else '❌'}",
             "",
             "> Each `carry-in` is read by ONE softmax attention from the nearest "
             "non-9 column below — the bright cell in its row. A run of 9s (PROP) "
             "is skipped, so the query hops multiple columns: that hop is the carry chain."]
    return fig, "\n".join(rows)


# ---- faithfulness bar chart ----
def faithfulness_chart():
    data = latest("faithfulness.json")
    if data is None:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.text(0.5, 0.5, "Run main.py.", ha="center"); ax.axis("off")
        return fig
    sweep = data["carry_sweep"]
    full = [data["full_em_mean"][str(k)] for k in sweep]
    abl = [data["ablated_em_mean"][str(k)] for k in sweep]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(sweep)); w = 0.38
    ax.bar([i - w / 2 for i in x], full, w, label="full (carry attention on)", color="#2a7ae2")
    ax.bar([i + w / 2 for i in x], abl, w, label="carry-attention ablated (= linear baseline)", color="#d1495b")
    ax.set_xticks(list(x)); ax.set_xticklabels([f"{k} carries" for k in sweep])
    ax.set_ylabel("exact-match rate"); ax.set_ylim(0, 1.05)
    ax.set_title(f"Carry-lookahead attention is causal\n(mean over {len(data['seeds'])} held-out seeds)")
    ax.legend(loc="lower left", fontsize=8)
    for i, v in enumerate(full):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(abl):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


# ---- operating-range line chart over digit width ----
def generalization_chart():
    data = latest("generalization.json")
    if data is None:
        fig, ax = plt.subplots(figsize=(7, 4)); ax.text(0.5, 0.5, "Run main.py.", ha="center"); ax.axis("off")
        return fig
    D = [r["digits"] for r in data]
    em = [r["exact_match"] for r in data]
    base = [r["baseline_exact_match"] for r in data]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(D, em, "o-", color="#2a7ae2", label="carry-lookahead attention")
    ax.plot(D, base, "s--", color="#d1495b", label="linear (no-carry) baseline")
    ax.set_xlabel("operand digit width  (carry-chain length)")
    ax.set_ylabel("exact-match rate"); ax.set_ylim(-0.03, 1.05)
    ax.set_title("Operating range: same attention, operands up to 10¹²")
    ax.axvline(3, color="gray", ls=":", lw=1); ax.text(3.05, 0.5, "canonical (3)", fontsize=8, color="gray")
    ax.legend(loc="center right", fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def summary_md():
    f = latest("faithfulness.json")
    if f is None:
        return "No run yet — execute `main.py`."
    return (f"**carry_robustness** — full: **{f['carry_robustness_full']:.3f}**  |  "
            f"ablated: **{f['carry_robustness_ablated']:.3f}** "
            f"(seeds {f['seeds'][0]}–{f['seeds'][-1]}). The carry chain is a single "
            "softmax attention, not a loop; ablate it and the carrying slices die.")


# ==========================================================================
with gr.Blocks(title="attention_int_add — pass_3") as demo:
    gr.Markdown("# attention_int_add — pass_3: carry propagation **as attention**")
    gr.Markdown(
        "Carry-lookahead adder. Each answer column's carry is read by ONE softmax "
        "attention from the nearest non-9 column below it — the whole carry chain "
        "resolved in parallel, no column-by-column ripple loop. Hand-set weights; "
        "`base_model.py` + value-embedding + 2 hand-set attention layers."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Probe the carry-lookahead attention")
                    a_in = gr.Number(label="a (0–999)", value=999, precision=0)
                    b_in = gr.Number(label="b (0–999)", value=1, precision=0)
                    run_btn = gr.Button("Compute", variant="primary")
                    with gr.Row():
                        p1 = gr.Button("999 + 1")
                        p2 = gr.Button("99 + 901")
                        p3 = gr.Button("500 + 500")
                        p4 = gr.Button("123 + 456")
                    heat = gr.Plot()
                with gr.Column(scale=1):
                    probe_md = gr.Markdown()
            gr.Markdown("---")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Faithfulness: ablate the carry-attention layer")
                    summ = gr.Markdown()
                    fchart = gr.Plot()
                with gr.Column():
                    gr.Markdown("### Operating range: carry-chain length")
                    gchart = gr.Plot()
        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)

    run_btn.click(probe, inputs=[a_in, b_in], outputs=[heat, probe_md])

    def _preset(a, b):
        h, m = probe(a, b)
        return a, b, h, m

    p1.click(lambda: _preset(999, 1), outputs=[a_in, b_in, heat, probe_md])
    p2.click(lambda: _preset(99, 901), outputs=[a_in, b_in, heat, probe_md])
    p3.click(lambda: _preset(500, 500), outputs=[a_in, b_in, heat, probe_md])
    p4.click(lambda: _preset(123, 456), outputs=[a_in, b_in, heat, probe_md])

    def _init():
        h, m = probe(999, 1)
        return h, m, summary_md(), faithfulness_chart(), generalization_chart()

    demo.load(_init, outputs=[heat, probe_md, summ, fchart, gchart])


if __name__ == "__main__":
    demo.launch()
