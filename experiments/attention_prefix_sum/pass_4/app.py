"""Gradio app for attention_prefix_sum / pass_4.

Demo tab: the "clock" prefix-sum circuit.
  * Operating-range line chart   — accuracy vs seq_len, full vs ablations.
  * Causal-evidence bar chart    — accuracy at L=16, full vs ablations vs random.
  * Attention-pattern heatmap    — the triangular prefix mask the head uses.
  * Interactive trace            — type a sequence, see prefix sum, target, and
                                   the circuit's prediction (numpy mirror of the
                                   exact same math run on GPU in main.py).
Benchmark tab: agentic.experiments.benchmark_panel over the whole goal.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

V = 10
SWEEP = [4, 8, 16, 32, 64]
ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS_ROOT = ATTEMPT_DIR / "results"


# --------------------------------------------------------------------------- #
# Run discovery / loading
# --------------------------------------------------------------------------- #
def _run_names() -> list[str]:
    if not RESULTS_ROOT.exists():
        return []
    return sorted(d.name for d in RESULTS_ROOT.iterdir() if d.is_dir())


def _resolve_run(run_name: str | None) -> Path | None:
    names = _run_names()
    if not names:
        return None
    if run_name and run_name in names:
        return RESULTS_ROOT / run_name
    return RESULTS_ROOT / names[-1]


def _load_ablations(run_name: str | None):
    run = _resolve_run(run_name)
    if run is None:
        return None
    p = run / "ablations.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _load_attention(run_name: str | None):
    run = _resolve_run(run_name)
    if run is None:
        return None
    p = run / "attention_L16.npy"
    if not p.exists():
        return None
    return np.load(p)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _empty_fig(msg: str):
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11)
    ax.axis("off")
    return fig


def operating_fig(run_name):
    abl = _load_ablations(run_name)
    if abl is None:
        return _empty_fig("No runs yet — run main.py first.")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for label, accs in abl.items():
        ys = [accs.get(str(L), accs.get(L, 0.0)) for L in SWEEP]
        style = "-o" if label.startswith("full") else "--o"
        ax.plot(SWEEP, ys, style, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SWEEP)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("sequence length (log scale, 2 orders of magnitude)")
    ax.set_ylabel("prefix-sum accuracy")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Operating range: clock circuit holds; ablations decay")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def bar_fig(run_name):
    abl = _load_ablations(run_name)
    if abl is None:
        return _empty_fig("No runs yet — run main.py first.")
    labels, vals = [], []
    for label, accs in abl.items():
        labels.append(label.split(" (")[0])
        vals.append(accs.get("16", accs.get(16, 0.0)))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["#2a9d8f", "#e76f51", "#f4a261", "#999999"][: len(labels)]
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy @ L=16 (canonical)")
    ax.set_title("Causal evidence: knock out a piece, behaviour breaks")
    ax.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.tight_layout()
    return fig


def attention_fig(run_name):
    W = _load_attention(run_name)
    if W is None:
        return _empty_fig("No runs yet — run main.py first.")
    fig, ax = plt.subplots(figsize=(5, 4.4))
    im = ax.imshow(W, cmap="viridis", aspect="equal")
    ax.set_xlabel("key position j")
    ax.set_ylabel("query position i")
    ax.set_title("Attention head: W[i,j] = 1/(i+1) for j<=i\n(the triangular prefix mask)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Interactive trace — numpy mirror of the GPU circuit (same exact math)
# --------------------------------------------------------------------------- #
def _predict_np(tokens: np.ndarray):
    x = tokens.astype(np.float64)
    S = np.round(np.cumsum(x))                                # causal accumulation
    f = np.arange(V)
    c = np.arange(V)
    ang = 2 * np.pi * np.outer(S, f) / V
    a, b = np.cos(ang), np.sin(ang)
    w_cos = np.cos(2 * np.pi * np.outer(f, c) / V)
    w_sin = np.sin(2 * np.pi * np.outer(f, c) / V)
    logits = a @ w_cos + b @ w_sin
    pred = logits.argmax(axis=1)
    target = (np.cumsum(tokens) % V)
    return pred, S.astype(int), target


def trace(token_str):
    try:
        toks = [int(t) for t in token_str.replace(",", " ").split()]
        toks = [max(0, min(V - 1, t)) for t in toks]
    except Exception:
        toks = []
    if not toks:
        return [["—", "—", "—", "—", "—"]], "Enter integers 0–9, e.g. `3, 1, 4, 1, 5, 9, 2`."
    tokens = np.array(toks, dtype=np.int64)
    pred, S, target = _predict_np(tokens)
    rows = []
    n_ok = 0
    for i, t in enumerate(toks):
        ok = pred[i] == target[i]
        n_ok += int(ok)
        rows.append([i, t, int(S[i]), int(target[i]),
                     f"{int(pred[i])} {'✓' if ok else '✗'}"])
    status = (f"Circuit matched the prefix-sum target at **{n_ok}/{len(toks)}** "
              f"positions. The clock readout maps the (possibly large) cumulative "
              f"sum S onto (S mod {V}) exactly.")
    return rows, status


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
_RUNS = _run_names()
_DEFAULT_RUN = _RUNS[-1] if _RUNS else None
_DEFAULT_TOKENS = "3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3"
_TABLE_HEADERS = ["pos i", "token x_i", "cumsum S_i", f"target (S mod {V})", "circuit pred"]

with gr.Blocks(title="Attention Prefix Sum — Clock Circuit") as demo:
    gr.Markdown(
        "# Attention Prefix Sum — the *clock* circuit\n"
        "A single hand-built attention head computes the **prefix sum**, and a "
        "fixed **Fourier (clock) unembedding** performs the **mod V** — the same "
        "representation real transformers grok for modular arithmetic. No MLP, "
        "no recurrence, no `remainder` call."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            run_dd = gr.Dropdown(
                choices=_RUNS, value=_DEFAULT_RUN, label="results run",
                interactive=True,
            )
            with gr.Row():
                op_plot = gr.Plot(label="Operating range")
                bar_plot = gr.Plot(label="Ablations @ L=16")
            attn_plot = gr.Plot(label="Attention pattern")

            gr.Markdown("### Interactive trace — type a token sequence")
            tok_in = gr.Textbox(value=_DEFAULT_TOKENS, label="tokens (0–9)")
            run_btn = gr.Button("Trace circuit", variant="primary")
            status_md = gr.Markdown()
            table_out = gr.Dataframe(headers=_TABLE_HEADERS, label="per-position trace")

            run_dd.change(operating_fig, inputs=run_dd, outputs=op_plot)
            run_dd.change(bar_fig, inputs=run_dd, outputs=bar_plot)
            run_dd.change(attention_fig, inputs=run_dd, outputs=attn_plot)

            run_btn.click(trace, inputs=tok_in, outputs=[table_out, status_md])
            tok_in.submit(trace, inputs=tok_in, outputs=[table_out, status_md])

            demo.load(operating_fig, inputs=run_dd, outputs=op_plot)
            demo.load(bar_fig, inputs=run_dd, outputs=bar_plot)
            demo.load(attention_fig, inputs=run_dd, outputs=attn_plot)
            demo.load(trace, inputs=tok_in, outputs=[table_out, status_md])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
