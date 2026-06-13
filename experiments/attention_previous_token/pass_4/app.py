"""Gradio app for the relative-position-bias previous-token head.

Demo tab:
  (1) heatmap of the attention matrix on the clean canonical sequence -- a
      previous-token head shows a single bright sub-diagonal band;
  (2) noise robustness: previous-token mass vs noise against the uniform
      baseline and the self / two-back distractors;
  (3) causal ablation bar chart: prev / self / two-back mass as the bias center
      is moved 0->3, showing the previous-token mass peaks exactly at center 1.
Benchmark tab: shared leaderboard across attempts.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"

ALPHA = 8.0
CENTER = 1.0


def _causal_softmax(logits: np.ndarray) -> np.ndarray:
    L = logits.shape[0]
    mask = np.triu(np.ones((L, L), bool), k=1)
    x = np.where(mask, -1e30, logits.astype(np.float64))
    x = x - x.max(1, keepdims=True)
    e = np.where(mask, 0.0, np.exp(x))
    return e / np.clip(e.sum(1, keepdims=True), 1e-12, None)


def attention_heatmap():
    L = 64
    idx = np.arange(L)
    offset = idx[:, None] - idx[None, :]
    logits = -ALPHA * (offset - CENTER) ** 2
    attn = _causal_softmax(logits)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(attn, cmap="magma", aspect="auto")
    ax.set_title("Attention (clean) -- bright sub-diagonal = prev-token")
    ax.set_xlabel("key j")
    ax.set_ylabel("query i")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def list_runs() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    runs = [d for d in RESULTS_DIR.iterdir() if (d / "benchmark.json").exists()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def _load(run_name: str):
    for r in list_runs():
        if r.name == run_name:
            payload = json.loads((r / "benchmark.json").read_text())
            comp_path = r / "comparison.json"
            comp = json.loads(comp_path.read_text()) if comp_path.exists() else None
            return payload, comp
    return None, None


def render(run_name: str):
    payload, comp = _load(run_name)
    if payload is None:
        return "No run found.", gr.update(value=None), None, None
    sweep = payload["sweep"]
    base = payload["uniform_baseline"]
    canon = next(r for r in sweep if r["noise"] == payload["canonical_noise"])
    straw = (comp or {}).get("strawman_canonical", {})
    md = (
        "### Relative-position-bias previous-token head\n"
        f"- **prev_token_attn_canonical:** {canon['prev_token_attention']:.4f}\n"
        f"- **uniform baseline:** {base:.4f}  "
        f"(lift ratio {canon['prev_token_attention'] / base:.1f}x)\n"
        f"- **uniform strawman (measured):** "
        f"{straw.get('prev_token_attention', float('nan')):.4f}\n"
        f"- **self / two-back distractors:** "
        f"{canon['self_attention']:.4f} / {canon['two_back_attention']:.4f}\n"
        f"- **robustness @ max noise:** "
        f"{sweep[-1]['prev_token_attention'] / max(canon['prev_token_attention'], 1e-9):.2f}"
    )
    rows = [[r["noise"], round(r["prev_token_attention"], 4),
             round(r["self_attention"], 4), round(r["two_back_attention"], 4)] for r in sweep]
    table = {"headers": ["noise", "prev", "self", "two_back"], "data": rows}

    # Noise robustness plot.
    fig1, ax = plt.subplots(figsize=(5.5, 4))
    xs = [r["noise"] for r in sweep]
    ax.plot(xs, [r["prev_token_attention"] for r in sweep], "o-", label="prev-token (i-1)")
    ax.plot(xs, [r["self_attention"] for r in sweep], "s--", label="self (i)")
    ax.plot(xs, [r["two_back_attention"] for r in sweep], "^--", label="two-back (i-2)")
    ax.axhline(base, color="gray", ls=":", label=f"uniform baseline ({base:.3f})")
    ax.set_xlabel("residual noise")
    ax.set_ylabel("attention mass")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.set_title("Robustness sweep")
    fig1.tight_layout()

    # Causal ablation: prev mass tracks the bias center.
    fig2, ax2 = plt.subplots(figsize=(5.5, 4))
    abl = (comp or {}).get("ablation_center")
    if abl:
        cs = [a["center"] for a in abl]
        w = 0.25
        ax2.bar([c - w for c in cs], [a["prev_token_attention"] for a in abl], w, label="prev (i-1)")
        ax2.bar([c for c in cs], [a["self_attention"] for a in abl], w, label="self (i)")
        ax2.bar([c + w for c in cs], [a["two_back_attention"] for a in abl], w, label="two-back (i-2)")
        ax2.set_xlabel("bias center (offset i-j)")
        ax2.set_ylabel("attention mass @ noise 0")
        ax2.set_title("Causal ablation: move the bias, the peak follows")
        ax2.legend(fontsize=8)
    else:
        ax2.text(0.5, 0.5, "no comparison.json", ha="center")
    fig2.tight_layout()

    return md, gr.update(value=table), fig1, fig2


_runs = [r.name for r in list_runs()]
_default = _runs[0] if _runs else None

with gr.Blocks(title="Previous-token head (relative-position bias)") as demo:
    gr.Markdown(
        "# Previous-token head via relative-position bias\n"
        "Hand-set, content-blind attention bias `logits[i,j] = -alpha*((i-j)-1)^2`, "
        "peaked at offset 1 so query `i` attends to key `i-1`. It is the minimal "
        "delta from `base_model.py` attention (one additive positional bias, no "
        "MLP, one head) and, because it ignores content, holds under any noise. "
        "All compute runs on CUDA."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(choices=_runs, value=_default, label="run")
            summary = gr.Markdown()
            with gr.Row():
                heat = gr.Plot(label="Attention matrix (clean)")
                sweep_plot = gr.Plot(label="Noise robustness")
            ablation_plot = gr.Plot(label="Causal ablation (bias center sweep)")
            sweep_tbl = gr.Dataframe(
                headers=["noise", "prev", "self", "two_back"], label="Per-noise metrics"
            )
            run_dd.change(render, inputs=run_dd,
                          outputs=[summary, sweep_tbl, sweep_plot, ablation_plot])
            demo.load(render, inputs=run_dd,
                      outputs=[summary, sweep_tbl, sweep_plot, ablation_plot])
            demo.load(attention_heatmap, inputs=None, outputs=heat)
        with gr.Tab("Benchmark"):
            benchmark_panel(str(GOAL_DIR)).render()


if __name__ == "__main__":
    demo.launch()
