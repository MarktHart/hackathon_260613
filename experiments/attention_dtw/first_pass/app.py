"""Gradio app for attention_dtw / first_pass.

Demo tab: for a chosen warp, show the content-matching head's attention heatmap
(key n -> query m) with the ground-truth warp path overlaid, next to the
diagonal strawman. A bar/line chart shows best-head vs. diagonal overlap across
the warp sweep — the core "alignment circuit, not a diagonal shortcut" claim.

Benchmark tab: the shared cross-attempt panel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

HERE = Path(__file__).resolve().parent
GOAL_DIR = HERE.parent
RESULTS = HERE / "results"

HEAD_LABELS = {
    "content_l2": "content (L2 distance) — alignment circuit",
    "content_dot": "content (dot product)",
    "diagonal": "diagonal — positional strawman",
}


def list_runs():
    if not RESULTS.exists():
        return []
    runs = sorted((p.name for p in RESULTS.iterdir() if p.is_dir()), reverse=True)
    return runs


def _load(run_id):
    run_dir = RESULTS / run_id
    with open(run_dir / "benchmark.json") as f:
        bench = json.load(f)
    npz = np.load(run_dir / "demo_examples.npz", allow_pickle=True)
    return bench, npz


def _payload(bench):
    # record_benchmark stores the payload; tolerate a couple of shapes.
    if "payload" in bench:
        return bench["payload"]
    if "sweep" in bench:
        return bench
    # nested under e.g. "result"
    for v in bench.values():
        if isinstance(v, dict) and "sweep" in v:
            return v
    return bench


def heatmaps(run_id, warp_key):
    bench, npz = _load(run_id)
    head_names = [str(x) for x in npz["head_names"]]
    attn = npz[f"{warp_key}__attn"]   # (H, N, M)
    align = npz[f"{warp_key}__align"]  # (N,)
    H, N, M = attn.shape

    fig, axes = plt.subplots(1, H, figsize=(4.2 * H, 4.6))
    if H == 1:
        axes = [axes]
    for h, ax in enumerate(axes):
        ax.imshow(attn[h], aspect="auto", origin="lower",
                  cmap="magma", vmin=0, vmax=1)
        # ground-truth warp path: query index (x) vs key index (y)
        ax.plot(align, np.arange(N), "-", color="#39ff14", lw=1.6,
                label="ground-truth warp")
        ax.scatter(np.argmax(attn[h], axis=1), np.arange(N), s=10,
                   color="cyan", label="argmax")
        name = head_names[h]
        ax.set_title(HEAD_LABELS.get(name, name), fontsize=9)
        ax.set_xlabel("query position m")
        if h == 0:
            ax.set_ylabel("key position n")
        ax.legend(loc="lower right", fontsize=6)
    fig.suptitle(f"warp = {warp_key}   (attn[n, m]; bright = high weight)", fontsize=11)
    fig.tight_layout()
    return fig


def overlap_curve(run_id):
    bench, _ = _load(run_id)
    pl = _payload(bench)
    warps = [s["warp"] for s in pl["sweep"]]
    best = [s["best_head_overlap"] for s in pl["sweep"]]
    meanh = [s["mean_head_overlap"] for s in pl["sweep"]]
    diag = [b["diagonal_overlap"] for b in pl["baseline"]]
    uni = [b["uniform_overlap"] for b in pl["baseline"]]

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(warps, best, "o-", color="#39ff14", lw=2.2, label="best head (content)")
    ax.plot(warps, meanh, "s--", color="#7fd4ff", lw=1.4, label="mean over heads")
    ax.plot(warps, diag, "^-", color="#ff6b6b", lw=2.0, label="diagonal baseline")
    ax.plot(warps, uni, ":", color="gray", lw=1.2, label="chance (1/M)")
    ax.set_xlabel("warp")
    ax.set_ylabel("path overlap (argmax == ground truth)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Alignment retained under warp: content head vs. diagonal")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def summary(run_id):
    bench, _ = _load(run_id)
    pl = _payload(bench)
    sw = {s["warp"]: s for s in pl["sweep"]}
    bl = {b["warp"]: b for b in pl["baseline"]}
    canon = pl["canonical_warp"]
    lo = pl["sweep"][0]["best_head_overlap"]
    hi = pl["sweep"][-1]["best_head_overlap"]
    rob = max(0.0, min(1.0, hi / lo if lo > 1e-12 else 0.0))
    c = sw.get(canon, {})
    cd = bl.get(canon, {})
    return (
        f"### Run `{run_id}`\n"
        f"- **alignment_robustness (headline):** {rob:.3f}  "
        f"(overlap retained at warp {pl['warp_sweep'][-1]:g} vs {pl['warp_sweep'][0]:g})\n"
        f"- **path_overlap_canonical** (warp {canon:g}): {c.get('best_head_overlap', 0):.3f}\n"
        f"- diagonal baseline at canonical: {cd.get('diagonal_overlap', 0):.3f}  "
        f"→ **lift over diagonal:** "
        f"{c.get('best_head_overlap', 0) - cd.get('diagonal_overlap', 0):+.3f}\n"
        f"- monotonicity at canonical: {c.get('monotonicity', 0):.3f}\n"
        f"- heads: {pl['num_heads']}"
    )


def warp_keys(run_id):
    if not run_id:
        return gr.update(choices=[], value=None)
    _, npz = _load(run_id)
    warps = [f"{w:g}" for w in npz["warps"]]
    # default to canonical 0.5 if present
    default = "0.5" if "0.5" in warps else warps[0]
    return gr.update(choices=warps, value=default)


def refresh(run_id, warp_key):
    if not run_id:
        return None, None, "No runs found — run main.py first."
    if warp_key is None:
        _, npz = _load(run_id)
        warps = [f"{w:g}" for w in npz["warps"]]
        warp_key = "0.5" if "0.5" in warps else warps[0]
    return heatmaps(run_id, warp_key), overlap_curve(run_id), summary(run_id)


_runs = list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks(title="attention_dtw / first_pass") as demo:
    gr.Markdown(
        "# attention_dtw — content-matching alignment head\n"
        "A hand-built attention head that aligns a time-warped key sequence to "
        "its source **queries** by feature content. It tracks the ground-truth "
        "warp path while a diagonal (position-only) head degrades as the warp grows."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(_runs, value=_default, label="run")
                warp_dd = gr.Dropdown([], label="warp slice")
            md = gr.Markdown()
            heat = gr.Plot(label="attention heatmaps + warp path")
            curve = gr.Plot(label="overlap vs warp")

            run_dd.change(warp_keys, inputs=run_dd, outputs=warp_dd)
            run_dd.change(refresh, inputs=[run_dd, warp_dd],
                          outputs=[heat, curve, md])
            warp_dd.change(refresh, inputs=[run_dd, warp_dd],
                           outputs=[heat, curve, md])

            def _init(run_id):
                upd = warp_keys(run_id)
                wk = upd.get("value") if isinstance(upd, dict) else None
                h, c, m = refresh(run_id, wk)
                return upd, h, c, m

            demo.load(_init, inputs=run_dd, outputs=[warp_dd, heat, curve, md])

        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
