"""Gradio app for attention_viterbi / pass_2 (hand-built predecessor head)."""
import json
from pathlib import Path

import numpy as np
import gradio as gr

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"


def list_runs():
    if not RESULTS_DIR.exists():
        return []
    return sorted([d.name for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)


def _load(run):
    base = RESULTS_DIR / run
    out = {}
    for key, name in (("benchmark", "benchmark.json"), ("payload", "payload.json"),
                      ("artifacts", "artifacts.json")):
        p = base / name
        out[key] = json.load(open(p)) if p.exists() else {}
    ap = base / "attn_weights.npy"
    out["attn"] = np.load(ap) if ap.exists() else None
    return out


def _blank(msg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    return fig


def fig_per_head(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ph = d["payload"].get("per_head", [])
    if not ph:
        return _blank("no per_head data")
    best = d["payload"].get("best_head", {})
    labels = [f"L{r['layer']}H{r['head']}" for r in ph]
    vals = [r["excess"] for r in ph]
    blabel = f"L{best.get('layer')}H{best.get('head')}"
    fig, ax = plt.subplots(figsize=(8, 3.4))
    colors = ["#d62728" if lab == blabel else "#1f77b4" for lab in labels]
    ax.bar(labels, vals, color=colors, edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("excess on t-1")
    ax.set_title("Per-head Viterbi excess  (red = best; 0 = uniform reader)")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    fig.tight_layout()
    return fig


def fig_positional(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = d["payload"].get("positional", [])
    if not pos:
        return _blank("no positional data")
    xs = [p["pos"] for p in pos]
    ys = [p["excess"] for p in pos]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(xs, ys, "o-", color="#d62728", label="best head")
    ax.axhline(0, color="k", lw=0.8, label="uniform baseline (0)")
    ax.set_xlabel("query position t")
    ax.set_ylabel("excess on t-1")
    ax.set_title("Excess by query position (best head)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_ablation(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ab = d["artifacts"].get("ablation", [])
    if not ab:
        return _blank("no ablation data")
    labels = [r["label"] for r in ab]
    vals = [r["headline"] for r in ab]
    colors = []
    for lab in labels:
        if lab.startswith("full"):
            colors.append("#2ca02c")
        elif "L0H0" in lab or "positional" in lab:
            colors.append("#d62728")   # the knockouts that should kill the signal
        else:
            colors.append("#7f7f7f")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(labels, vals, color=colors, edgecolor="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("headline excess (max over heads)")
    ax.set_title("Causal localization: only L0H0 / positional-enc carry the signature")
    fig.tight_layout()
    return fig


def fig_operating(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    art = d["artifacts"]
    seeds = art.get("seed_sweep", [])
    temps = art.get("temp_sweep", [])
    seqs = art.get("seqlen_sweep", [])
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    if seeds:
        axes[0].bar([str(s["seed"]) for s in seeds], [s["excess"] for s in seeds], color="#1f77b4")
        axes[0].set_title("across HMM seed")
        axes[0].set_xlabel("seed"); axes[0].set_ylabel("excess (best head)")
        axes[0].axhline(0, color="k", lw=0.6)
    if temps:
        axes[1].plot([t["temp"] for t in temps], [t["excess"] for t in temps], "o-", color="#ff7f0e")
        axes[1].set_xscale("symlog")
        axes[1].set_title("across attention sharpness")
        axes[1].set_xlabel("temperature (symlog)"); axes[1].set_ylabel("excess")
        axes[1].axhline(0, color="k", lw=0.6)
    if seqs:
        axes[2].plot([s["T"] for s in seqs], [s["excess"] for s in seqs], "o-", color="#2ca02c", label="excess")
        axes[2].plot([s["T"] for s in seqs], [s["robustness"] for s in seqs], "s--", color="#9467bd", label="robustness")
        axes[2].set_title("across sequence length")
        axes[2].set_xlabel("seq len T"); axes[2].legend(fontsize=8)
        axes[2].axhline(0, color="k", lw=0.6)
    fig.tight_layout()
    return fig


def fig_heatmap(d, layer, head, seq):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    attn = d["attn"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if attn is None:
        return _blank("attn_weights.npy missing")
    seq = min(int(seq), attn.shape[0] - 1)
    mat = attn[int(seq), int(layer), int(head)]
    im = ax.imshow(mat, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"L{int(layer)}H{int(head)} seq#{seq}\n(red diag-below = attends t-1)")
    ax.set_xlabel("key s"); ax.set_ylabel("query t")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def render(run):
    if not run:
        b = _blank("no runs yet")
        return b, b, b, b, b, "{}"
    d = _load(run)
    metrics = json.dumps(d["benchmark"], indent=2)
    best = d["payload"].get("best_head", {"layer": 0, "head": 0})
    return (
        fig_per_head(d), fig_positional(d), fig_ablation(d), fig_operating(d),
        fig_heatmap(d, best.get("layer", 0), best.get("head", 0), 0), metrics,
    )


def render_heatmap(run, layer, head, seq):
    if not run:
        return _blank("no runs yet")
    return fig_heatmap(_load(run), layer, head, seq)


with gr.Blocks(title="attention_viterbi — pass_2") as demo:
    gr.Markdown(
        "# attention_viterbi — pass_2 (hand-built predecessor head)\n"
        "A 2L/4H/d64 attention-only transformer with **hand-set weights**. Layer-0 head-0 is "
        "built to attend to query position **t-1** — the Viterbi backpointer for a first-order HMM. "
        "Excess = `α[t,t-1] − mean(α[t,:t])`; a uniform reader scores 0."
    )
    runs = list_runs()
    default_run = runs[0] if runs else None

    with gr.Tab("Demo"):
        run_dd = gr.Dropdown(choices=runs, value=default_run, label="Run", interactive=True)
        with gr.Row():
            per_head = gr.Plot(label="Per-head excess")
            positional = gr.Plot(label="Excess by position (best head)")
        gr.Markdown("**Faithfulness (causal):** knock out one head's attention (replace with uniform) "
                    "or zero the positional encoding, then re-measure. Only L0H0 / the positional "
                    "code carries the Viterbi signature.")
        ablation = gr.Plot(label="Ablation / strawman")
        gr.Markdown("**Operating range:** the mechanism holds across HMM seeds and sequence length, "
                    "and sharpens monotonically with attention temperature (collapsing to 0 at temp→0).")
        operating = gr.Plot(label="Operating range")
        gr.Markdown("**Raw attention** — pick a head/sequence; the predecessor head shows a bright "
                    "sub-diagonal (mass on key t-1).")
        with gr.Row():
            layer_sl = gr.Slider(0, 1, step=1, value=0, label="layer")
            head_sl = gr.Slider(0, 3, step=1, value=0, label="head")
            seq_sl = gr.Slider(0, 29, step=1, value=0, label="sequence #")
        heatmap = gr.Plot(label="Attention heatmap")
        metrics = gr.Code(label="benchmark.json", language="json")

        outs = [per_head, positional, ablation, operating, heatmap, metrics]
        run_dd.change(render, inputs=[run_dd], outputs=outs)
        for ctrl in (layer_sl, head_sl, seq_sl):
            ctrl.change(render_heatmap, inputs=[run_dd, layer_sl, head_sl, seq_sl], outputs=[heatmap])
        demo.load(render, inputs=[run_dd], outputs=outs)

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()
