"""Demo + Benchmark app for the hand-built Dyck stack-matching circuit.

Demo tab: for a chosen test sequence and head, render the post-softmax
attention matrix (query rows x key cols) over the bracket string, with the
ground-truth matching-open position marked by a ring on each closing-bracket
row. Head 0 is the matcher (rings land on the bright cell); head 1 is the depth
code (mass increases with nesting depth).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

TOK = {0: "·", 1: "(", 2: ")", 3: "B", 4: "E"}
HEAD_NAMES = {0: "head 0 — matcher", 1: "head 1 — depth code"}


def _runs():
    if not RESULTS.is_dir():
        return []
    return sorted([p.name for p in RESULTS.iterdir() if (p / "viz.npz").exists()], reverse=True)


def _load(run):
    npz = np.load(RESULTS / run / "viz.npz")
    return {k: npz[k] for k in npz.files}


def _metrics(run):
    bj = RESULTS / run / "benchmark.json"
    if not bj.exists():
        return {}
    return json.loads(bj.read_text()).get("metrics", {})


def render(run, ex_idx, head):
    if not run:
        return None, "No runs found — run main.py first."
    data = _load(run)
    ex_idx = int(ex_idx)
    head = int(head)
    L = int(data["lengths"][ex_idx])
    ids = data["input_ids"][ex_idx][:L]
    attn = data["attn"][ex_idx][head][:L, :L]
    match = data["matching_open_pos"][ex_idx][:L]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(attn, cmap="magma", vmin=0, vmax=max(attn.max(), 1e-3), aspect="equal")
    labels = [TOK.get(int(t), "?") for t in ids]
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("key position (attended-to token)")
    ax.set_ylabel("query position (current token)")
    ax.set_title(f"{HEAD_NAMES[head]} — example {int(data['example_ids'][ex_idx])}")

    # Ring the ground-truth matching open for each closing-bracket row.
    for i in range(L):
        m = int(match[i])
        if m >= 0:
            ax.add_patch(
                plt.Circle((m, i), 0.42, fill=False, edgecolor="cyan", lw=1.6)
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="attention weight")
    fig.tight_layout()

    # Caption: does argmax match the cyan ring on closing rows?
    closes = [i for i in range(L) if int(ids[i]) == 2 and int(match[i]) >= 0]
    if head == 0 and closes:
        hits = sum(1 for i in closes if int(np.argmax(attn[i, :L])) == int(match[i]))
        cap = (
            f"Cyan ring = true matching '('. Head-0 argmax lands on the ring for "
            f"**{hits}/{len(closes)}** closing brackets in this string."
        )
    elif head == 1:
        cap = (
            "Head 1 spreads each closing row's mass over all prior opens, brighter "
            "on deeper-nested ones — that monotone depth code is what `depth_corr` scores."
        )
    else:
        cap = "Cyan ring = true matching '('."
    return fig, cap


def metrics_md(run):
    m = _metrics(run)
    if not m:
        return "No metrics."
    keys = [
        "dyck_matching_canonical",
        "linear_baseline_matching",
        "lift_over_baseline_matching",
        "dyck_depth_corr_canonical",
        "dyck_diag_frac_mean",
    ]
    rows = ["| metric | value |", "|---|---|"]
    for k in keys:
        if k in m:
            rows.append(f"| `{k}` | {m[k]:.4f} |")
    return "\n".join(rows)


def on_run_change(run):
    data = _load(run) if run else {"example_ids": np.array([0])}
    n = len(data["example_ids"])
    dd = gr.update(choices=list(range(n)), value=0)
    fig, cap = render(run, 0, 0)
    return dd, fig, cap, metrics_md(run)


_initial = _runs()

with gr.Blocks(title="attention_dyck · first_pass") as demo:
    gr.Markdown(
        "# Dyck stack-matching circuit (hand-built)\n"
        "A single attention layer, two heads. **Head 0** matches each closing "
        "bracket to its opening partner (depth-band score + recency tiebreak); "
        "**head 1** is a monotone nesting-depth code. Pick a sequence and head."
    )
    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(_initial, value=(_initial[0] if _initial else None), label="run")
            ex_dd = gr.Dropdown([0], value=0, label="example")
            head_radio = gr.Radio([0, 1], value=0, label="head (0=matcher, 1=depth)")
        plot = gr.Plot(label="attention")
        caption = gr.Markdown()
        metrics_box = gr.Markdown()

        run_dd.change(on_run_change, inputs=run_dd, outputs=[ex_dd, plot, caption, metrics_box])
        ex_dd.change(render, inputs=[run_dd, ex_dd, head_radio], outputs=[plot, caption])
        head_radio.change(render, inputs=[run_dd, ex_dd, head_radio], outputs=[plot, caption])
        demo.load(on_run_change, inputs=run_dd, outputs=[ex_dd, plot, caption, metrics_box])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
