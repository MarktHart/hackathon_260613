"""Gradio app for the feature-prototype attention-mode classifier (pass_4).

Demo tab:
  - browse any head (noise x mode x head-index), see its attention heatmap;
  - the 5 hand-named features for that head vs its mode's prototype;
  - the full classifier's mode probabilities AND a strawman's, side by side;
  - a robustness curve: full vs strawman accuracy across the noise sweep.

Benchmark tab:
  - the shared benchmark_panel across every attempt at this goal.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

# --- constants fixed by the goal (no need to import task to know these) ---
MODES = ["positional", "uniform", "diagonal", "induction", "previous_token"]
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
FEATURE_NAMES = ["key0_mass", "diag(i,i)", "next(i,i+1)", "prev(i,i-1)", "row_peak"]
# bands that drive the 4 spiked-mode logits (cols 0..3); col 4 (row_peak) shown
# for context. The full classifier picks the largest band if it beats TAU,
# otherwise calls `uniform`.
BAND_NAMES = ["key0_mass", "diag(i,i)", "next(i,i+1)", "prev(i,i-1)"]
BAND_MODE = ["positional", "diagonal", "induction", "previous_token"]
TAU = 0.18

ATTEMPT_DIR = Path(__file__).resolve().parent
GOAL_DIR = ATTEMPT_DIR.parent

_CACHE = {"data": None, "loaded": False}


def _load():
    """Lazily load demo.npz from the most recent run (None if unavailable)."""
    if _CACHE["loaded"]:
        return _CACHE["data"]
    _CACHE["loaded"] = True
    results = ATTEMPT_DIR / "results"
    data = None
    if results.is_dir():
        runs = sorted(p for p in results.iterdir() if p.is_dir())
        for run in reversed(runs):
            npz = run / "demo.npz"
            if npz.is_file():
                d = np.load(npz, allow_pickle=True)
                data = {k: d[k] for k in d.files}
                break
    _CACHE["data"] = data
    return data


def _empty_fig(msg):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", wrap=True)
    ax.axis("off")
    return fig


def _select_index(data, noise_val, mode_idx, head_idx):
    mask = np.isclose(data["noise"], noise_val) & (data["true_idx"] == mode_idx)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        return None
    return int(idxs[int(head_idx) % len(idxs)])


def update(noise_str, mode_name, head_idx):
    data = _load()
    if data is None:
        e = _empty_fig("No results yet — run main.py first.")
        return e, _empty_fig("No results yet."), {}, {}, "No results yet."

    noise_val = float(noise_str)
    mode_idx = MODES.index(mode_name)
    sel = _select_index(data, noise_val, mode_idx, head_idx)
    if sel is None:
        e = _empty_fig("No head for this selection.")
        return e, _empty_fig("n/a"), {}, {}, "No head for this selection."

    mat = data["matrices"][sel]
    feat = data["feats"][sel]
    full = data["full_probs"][sel]
    straw = data["straw_probs"][sel]

    # 1) attention heatmap
    fig_h, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=max(0.2, mat.max()))
    ax.set_title(f"attention  ({mode_name}, noise={noise_val})", fontsize=10)
    ax.set_xlabel("key  j"); ax.set_ylabel("query  i")
    fig_h.colorbar(im, ax=ax, fraction=0.046)
    fig_h.tight_layout()

    # 2) the 4 band features vs the uniform threshold TAU (the decision rule)
    bands = feat[:4]
    winner = int(np.argmax(bands))
    decided = BAND_MODE[winner] if bands[winner] > TAU else "uniform"
    fig_f, ax2 = plt.subplots(figsize=(4.8, 3.8))
    x = np.arange(len(BAND_NAMES))
    colors = ["#c0392b" if i == winner else "#3b7dd8" for i in range(4)]
    ax2.bar(x, bands, width=0.6, color=colors)
    ax2.axhline(TAU, color="#e08214", ls="--", lw=2, label=f"uniform threshold TAU={TAU}")
    ax2.set_xticks(x)
    ax2.set_xticklabels(BAND_NAMES, rotation=30, ha="right", fontsize=8)
    ax2.set_ylabel("row-averaged band mass")
    ax2.set_title(f"band rule -> '{decided}'  (peak={feat[4]:.2f})", fontsize=10)
    ax2.legend(fontsize=8)
    fig_f.tight_layout()

    full_d = {MODES[i]: float(full[i]) for i in range(len(MODES))}
    straw_d = {MODES[i]: float(straw[i]) for i in range(len(MODES))}

    pred = MODES[int(np.argmax(full))]
    ok = "correct" if pred == mode_name else "WRONG"
    gt = (f"Ground truth: **{mode_name}**  |  full prediction: **{pred}** ({ok})  "
          f"|  noise = {noise_val}")
    return fig_h, fig_f, full_d, straw_d, gt


def robustness_fig():
    data = _load()
    if data is None:
        return _empty_fig("No results yet — run main.py first.")
    noise = data["noise"]
    true_idx = data["true_idx"]
    full_pred = np.argmax(data["full_probs"], axis=1)
    straw_pred = np.argmax(data["straw_probs"], axis=1)

    levels = sorted(set(float(n) for n in noise))
    full_acc, straw_acc = [], []
    for lv in levels:
        m = np.isclose(noise, lv)
        full_acc.append(float((full_pred[m] == true_idx[m]).mean()))
        straw_acc.append(float((straw_pred[m] == true_idx[m]).mean()))

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(levels, full_acc, "o-", label="full (5 features)", color="#3b7dd8", lw=2)
    ax.plot(levels, straw_acc, "s--", label="strawman (3 bands only)", color="#c0392b", lw=2)
    ax.axhline(0.2, color="gray", ls=":", label="random (0.2)")
    ax.set_xlabel("noise level"); ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mode-naming accuracy vs corruption", fontsize=11)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown(
        "# Attention-Mode Classifier — band rule (pass_4)\n"
        "Hand-built: each head is read by named features (anchor-key mass and the "
        "diagonal / next / prev bands). A head is the mode whose band carries the "
        "most row-averaged mass **if that beats `TAU=0.18`**; otherwise it is "
        "`uniform`. This argmax-vs-threshold rule is scale-invariant, so it "
        "survives noise that merely shrinks every spike. The **strawman** instead "
        "matches absolute distance to the clean prototypes and collapses toward "
        "`uniform` as the spikes fade."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                noise_dd = gr.Dropdown(
                    choices=[str(n) for n in NOISE_LEVELS], value="0.0",
                    label="noise level")
                mode_dd = gr.Dropdown(choices=MODES, value="positional",
                                      label="ground-truth mode")
                head_sl = gr.Slider(0, 9, value=0, step=1, label="head index (0-9)")
            with gr.Row():
                heat = gr.Plot(label="attention heatmap")
                feats = gr.Plot(label="features vs prototype")
            gt_md = gr.Markdown()
            with gr.Row():
                full_lbl = gr.Label(num_top_classes=5, label="full classifier P(mode)")
                straw_lbl = gr.Label(num_top_classes=5, label="strawman P(mode)")
            gr.Markdown("### Robustness: full vs strawman across the noise sweep")
            robust = gr.Plot(label="accuracy vs noise")

            inputs = [noise_dd, mode_dd, head_sl]
            outs = [heat, feats, full_lbl, straw_lbl, gt_md]
            noise_dd.change(update, inputs=inputs, outputs=outs)
            mode_dd.change(update, inputs=inputs, outputs=outs)
            head_sl.change(update, inputs=inputs, outputs=outs)
            demo.load(update, inputs=inputs, outputs=outs)
            demo.load(robustness_fig, inputs=None, outputs=robust)

        with gr.Tab("Benchmark"):
            gr.Markdown("## Benchmark across all attempts at `attention_mode`")
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
