import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
RESULTS = Path(__file__).parent / "results"


def _run_names():
    if not RESULTS.exists():
        return []
    return sorted([d.name for d in RESULTS.iterdir() if d.is_dir()], reverse=True)


def _load(run_name, fname):
    if not run_name:
        return None
    p = RESULTS / run_name / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def plot_match_vs_uniform(run_name):
    bench = _load(run_name, "benchmark.json")
    if not bench:
        return None
    payload = bench.get("payload", bench)
    sweep = payload["sweep"]
    L = [r["L"] for r in sweep]
    mm = [r["match_mass"] for r in sweep]
    ub = [r["uniform_baseline"] for r in sweep]
    x = np.arange(len(L))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, mm, 0.4, label="equality head", color="#2E86AB")
    ax.bar(x + 0.2, ub, 0.4, label="uniform baseline", color="#F18F01")
    for xi, m in zip(x, mm):
        ax.text(xi - 0.2, m + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(L)
    ax.set_xlabel("sequence length L"); ax.set_ylabel("match_mass = attn[p2, p1]")
    ax.set_ylim(0, 1.08)
    ax.set_title("Equality routing vs uniform baseline (higher = real lookup)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ablations(run_name):
    abl = _load(run_name, "ablations.json")
    if not abl:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"full": "#2E86AB", "no_self_suppress": "#C0392B", "no_equality": "#7F8C8D"}
    labels = {
        "full": "full circuit",
        "no_self_suppress": "ablate self-suppression",
        "no_equality": "ablate QK equality",
    }
    L = [r["L"] for r in abl["full"]]
    for name, recs in abl.items():
        ax.plot(L, [r["match_mass"] for r in recs], "o-", color=colors.get(name),
                label=labels.get(name, name), linewidth=2, markersize=7)
    ax.plot(L, [r["uniform_baseline"] for r in abl["full"]], "k--",
            alpha=0.5, label="uniform baseline")
    ax.set_xscale("log", base=2); ax.set_xticks(L); ax.set_xticklabels(L)
    ax.set_xlabel("sequence length L"); ax.set_ylabel("match_mass = attn[p2, p1]")
    ax.set_ylim(0, 1.08)
    ax.set_title("Causal ablations: knock out a part, lookup breaks")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_heatmap(run_name):
    ex = _load(run_name, "real_example.json")
    if not ex:
        return None
    attn = np.array(ex["attn"]); L = ex["L"]; p1, p2 = ex["p1"], ex["p2"]
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(attn, cmap="viridis", origin="lower", aspect="auto", vmin=0, vmax=1)
    ax.axhline(p2, color="red", ls="--", lw=1, alpha=0.7)
    ax.axvline(p1, color="red", ls="--", lw=1, alpha=0.7)
    ax.plot(p1, p2, "r*", markersize=16)
    ax.set_xlabel("key position"); ax.set_ylabel("query position")
    ax.set_title(f"Real GPU attention (L={L})\nquery p2={p2} routes onto key p1={p1}")
    plt.colorbar(im, ax=ax, label="attention weight")
    fig.tight_layout()
    return fig


def refresh(run_name):
    run_name = run_name or (_run_names()[0] if _run_names() else None)
    return plot_match_vs_uniform(run_name), plot_ablations(run_name), plot_heatmap(run_name)


with gr.Blocks() as demo:
    gr.Markdown(
        "# Equality lookup head — pass_4 (hand-built, label-free)\n"
        "A single attention head with **one-hot token-identity** Q and K (so QKᵀ is high "
        "exactly where tokens match) plus a **uniform, position-agnostic** no-self-attention "
        "bias. For query `p2` the only earlier matching key is `p1`, so mass routes there by "
        "token matching alone — no p1/p2 oracle is used."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(label="run", choices=_run_names(),
                                     value=(_run_names()[0] if _run_names() else None))
                refresh_btn = gr.Button("Refresh", variant="secondary")
            with gr.Row():
                p_match = gr.Plot(label="Match mass vs uniform")
                p_abl = gr.Plot(label="Causal ablations (faithfulness)")
            p_heat = gr.Plot(label="Real attention heatmap")

            refresh_btn.click(lambda: gr.Dropdown(choices=_run_names(),
                              value=(_run_names()[0] if _run_names() else None)),
                              outputs=run_dd)
            run_dd.change(refresh, inputs=run_dd, outputs=[p_match, p_abl, p_heat])
            demo.load(refresh, inputs=run_dd, outputs=[p_match, p_abl, p_heat])

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Benchmark history across all attempts")
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
