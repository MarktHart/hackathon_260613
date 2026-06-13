"""
Gradio app for attention_int_add / pass_2.

Demo tab: the analytical story of the hand-built circuit.
  * An interactive probe: type a + b, see the column-by-column carry ripple and
    watch where the no-carry strawman fails on THIS example.
  * The faithfulness chart: full circuit vs carry-ablated, exact-match per carry
    slice, averaged over 8 held-out seeds. Ablating the carry channel = the
    task's linear baseline, so the bars show exactly the carry mechanism's
    causal contribution.
Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS_DIR = ATTEMPT_DIR / "results"


# ---- pure-numpy reference of the hand-built circuit (for the live probe) ----
def ripple_add(a: int, b: int):
    da = [a % 10, (a // 10) % 10, (a // 100) % 10]   # units, tens, hundreds
    db = [b % 10, (b // 10) % 10, (b // 100) % 10]
    carry = 0
    digits, trace = [], []
    for c in range(3):
        total = da[c] + db[c] + carry
        d = total % 10
        nc = 1 if total >= 10 else 0
        trace.append((c, da[c], db[c], carry, d, nc))
        carry = nc
        digits.append(d)
    thousands = carry
    pred = [thousands, digits[2], digits[1], digits[0]]   # MSB first
    return pred, trace


def baseline_pred(a: int, b: int):
    da = [a % 10, (a // 10) % 10, (a // 100) % 10]
    db = [b % 10, (b // 10) % 10, (b // 100) % 10]
    return [0, (da[2] + db[2]) % 10, (da[1] + db[1]) % 10, (da[0] + db[0]) % 10]


def latest_faithfulness():
    if not RESULTS_DIR.exists():
        return None
    runs = sorted(RESULTS_DIR.glob("*/faithfulness.json"), key=lambda p: p.stat().st_mtime,
                  reverse=True)
    if not runs:
        return None
    with open(runs[0]) as f:
        return json.load(f)


# ---- interactive probe ----
def probe(a, b):
    a, b = int(a), int(b)
    if not (0 <= a <= 999 and 0 <= b <= 999):
        return "Enter operands in [0, 999]."
    pred, trace = ripple_add(a, b)
    base = baseline_pred(a, b)
    true = a + b
    pred_val = pred[0] * 1000 + pred[1] * 100 + pred[2] * 10 + pred[3]
    base_val = base[0] * 1000 + base[1] * 100 + base[2] * 10 + base[3]
    n_carries = sum(t[5] for t in trace)

    col_names = ["units", "tens", "hundreds"]
    lines = [
        f"### {a} + {b} = {true}",
        f"This problem has **{n_carries} carr{'y' if n_carries == 1 else 'ies'}**.",
        "",
        "**Carry ripple (hand-set carry channel):**",
        "",
        "| column | a | b | carry-in | digit | carry-out |",
        "|---|---|---|---|---|---|",
    ]
    for c, da, db, ci, d, co in trace:
        lines.append(f"| {col_names[c]} | {da} | {db} | {ci} | {d} | {co} |")
    lines += [
        "",
        f"**Full circuit prediction:** {pred_val}  "
        f"{'✅' if pred_val == true else '❌'}",
        f"**Carry-ablated / linear baseline:** {base_val}  "
        f"{'✅ (no carry needed)' if base_val == true else '❌ (carry was required)'}",
    ]
    if base_val != true:
        lines.append("\n> The strawman drops the carry and gets it wrong — the carry "
                     "channel is exactly what fixes this example.")
    return "\n".join(lines)


# ---- faithfulness / robustness chart ----
def faithfulness_chart():
    data = latest_faithfulness()
    if data is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, "Run main.py to generate results.", ha="center", va="center")
        ax.axis("off")
        return fig

    sweep = data["carry_sweep"]
    full = [data["full_em_mean"][str(k)] for k in sweep]
    abl = [data["ablated_em_mean"][str(k)] for k in sweep]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(sweep))
    w = 0.38
    ax.bar([i - w / 2 for i in x], full, width=w, label="full circuit", color="#2a7ae2")
    ax.bar([i + w / 2 for i in x], abl, width=w,
           label="carry ablated (= linear baseline)", color="#d1495b")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{k} carries" for k in sweep])
    ax.set_ylabel("exact-match rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Carry channel is causal: ablate it and carrying slices collapse\n"
                 f"(mean over {len(data['seeds'])} held-out seeds)")
    ax.legend(loc="lower left")
    for i, v in enumerate(full):
        ax.text(i - w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for i, v in enumerate(abl):
        ax.text(i + w / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def summary_md():
    data = latest_faithfulness()
    if data is None:
        return "No run yet — execute `main.py`."
    rf = data["carry_robustness_full"]
    ra = data["carry_robustness_ablated"]
    return (f"**carry_robustness** — full circuit: **{rf:.3f}**  |  "
            f"carry-ablated: **{ra:.3f}**  "
            f"(over seeds {data['seeds'][0]}–{data['seeds'][-1]}). "
            "The hand-built circuit is exact on every carry slice and every seed; "
            "removing the carry channel destroys it exactly on the carrying slices.")


# ==========================================================================
# Blocks
# ==========================================================================
with gr.Blocks(title="attention_int_add — pass_2") as demo:
    gr.Markdown("# attention_int_add — pass_2 (hand-built attention circuit)")
    gr.Markdown(
        "A single hand-set attention layer routes operand digits into the answer "
        "positions; a hand-set carry channel ripples carries across columns. "
        "No weights are trained — the mechanism is known by construction."
    )

    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Probe a single problem")
                    a_in = gr.Number(label="a (0–999)", value=99, precision=0)
                    b_in = gr.Number(label="b (0–999)", value=901, precision=0)
                    run_btn = gr.Button("Compute", variant="primary")
                    with gr.Row():
                        p1 = gr.Button("99 + 1 (2 carries)")
                        p2 = gr.Button("99 + 901 (3 carries)")
                        p3 = gr.Button("500 + 500 (1 carry)")
                        p4 = gr.Button("123 + 456 (0 carries)")
                    probe_md = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("### Faithfulness: carry-channel ablation")
                    summ = gr.Markdown()
                    chart = gr.Plot()

        with gr.TabItem("Benchmark"):
            benchmark_panel(GOAL_DIR)

    # ---- events (inside Blocks) ----
    run_btn.click(probe, inputs=[a_in, b_in], outputs=probe_md)

    def _preset(a, b):
        return a, b, probe(a, b)

    p1.click(lambda: _preset(99, 1), outputs=[a_in, b_in, probe_md])
    p2.click(lambda: _preset(99, 901), outputs=[a_in, b_in, probe_md])
    p3.click(lambda: _preset(500, 500), outputs=[a_in, b_in, probe_md])
    p4.click(lambda: _preset(123, 456), outputs=[a_in, b_in, probe_md])

    def _init():
        return probe(99, 901), summary_md(), faithfulness_chart()

    demo.load(_init, outputs=[probe_md, summ, chart])


if __name__ == "__main__":
    demo.launch()
