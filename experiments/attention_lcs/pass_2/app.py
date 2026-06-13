import json
from pathlib import Path

import gradio as gr
import numpy as np

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"

HEAD_NAMES = ["token+pos (full)", "token only", "pos only", "uniform"]


def run_names():
    if not RESULTS.exists():
        return []
    return sorted([d.name for d in RESULTS.iterdir() if d.is_dir()], reverse=True)


def _load(run, fname):
    p = RESULTS / run / fname
    if p.exists():
        return json.loads(p.read_text())
    return None


def _fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _payload(run):
    bm = _load(run, "benchmark.json")
    if not bm:
        return None
    return bm.get("payload", bm)


def summary_md(run):
    bm = _payload(run)
    if not bm:
        return "No benchmark.json found for this run."
    base = bm["random_baseline_mass"]
    sweep = bm["sweep"]
    best = max(sweep, key=lambda r: r["lcs_lift"])
    headroom = 1.0 - base
    rob = max(0.0, min(1.0, best["lcs_lift"] / headroom)) if headroom > 0 else 0.0
    name = HEAD_NAMES[best["head"]] if best["head"] < len(HEAD_NAMES) else f"head {best['head']}"
    return (
        f"### Run `{run}`\n"
        f"- **Headline `lcs_lift_canonical`: {best['lcs_lift']:.3f}**  "
        f"(best head = head {best['head']}, *{name}*)\n"
        f"- best-head mass on LCS keys: **{best['lcs_attention_mass']:.3f}** "
        f"vs uniform baseline **{base:.3f}**\n"
        f"- `lcs_robustness` (lift / headroom): **{rob:.3f}**\n"
    )


def bar_plot(run):
    bm = _payload(run)
    if not bm:
        return None
    plt = _fig()
    sweep = sorted(bm["sweep"], key=lambda r: r["head"])
    base = bm["random_baseline_mass"]
    labels = [HEAD_NAMES[r["head"]] if r["head"] < len(HEAD_NAMES) else str(r["head"]) for r in sweep]
    mass = [r["lcs_attention_mass"] for r in sweep]
    colors = ["#1b7837", "#7fbf7b", "#af8dc3", "#999999"][: len(sweep)]

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bars = ax.bar(labels, mass, color=colors)
    ax.axhline(base, color="red", ls="--", lw=1.5, label=f"uniform baseline ({base:.3f})")
    for bar, m in zip(bars, mass):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("attention mass on LCS partner keys")
    ax.set_title("Ablation ladder: each removed component drops LCS mass")
    ax.set_ylim(0, max(1.0, max(mass) * 1.15))
    ax.legend()
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def heatmap(run):
    s = _load(run, "sample.json")
    if not s:
        return None
    plt = _fig()
    attn = np.array(s["attn_full"])
    mk = s["match_keys"]
    L = attn.shape[0]

    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    im = ax.imshow(attn, cmap="viridis", vmin=0, vmax=1, aspect="equal")
    # mark the true LCS partner cells
    for q, keys in enumerate(mk):
        for k in keys:
            ax.add_patch(plt.Rectangle((k - 0.5, q - 0.5), 1, 1,
                                       fill=False, edgecolor="red", lw=2.0))
    ax.set_xlabel("key position in B")
    ax.set_ylabel("query position in A")
    ax.set_title(f"Full head — red = true LCS partner (example #{s['index']})")
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    fig.colorbar(im, ax=ax, fraction=0.046, label="attention weight")
    plt.tight_layout()
    return fig


def range_plot(run):
    orr = _load(run, "operating_range.json")
    if not orr:
        return None
    plt = _fig()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(orr["vocab_sizes"], orr["lift"], "o-", color="#1b7837", label="head-0 lift")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("vocabulary size (log scale)")
    ax.set_ylabel("lcs_lift (mass - uniform)")
    ax.set_title("Operating range: sparser matches (big vocab) -> cleaner alignment")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig


def refresh(run):
    if not run:
        names = run_names()
        run = names[0] if names else None
    if not run:
        return "No runs found. Run main.py first.", None, None, None
    return summary_md(run), bar_plot(run), heatmap(run), range_plot(run)


with gr.Blocks() as demo:
    gr.Markdown("# attention_lcs — pass_2 (hand-built LCS-alignment head)")

    with gr.Tab("Demo"):
        names = run_names()
        with gr.Row():
            run_dd = gr.Dropdown(choices=names, value=names[0] if names else None,
                                 label="run")
            reload_btn = gr.Button("Reload", size="sm")
        md = gr.Markdown()
        with gr.Row():
            bars = gr.Plot(label="Per-head ablation ladder")
            hm = gr.Plot(label="Attention heatmap (full head)")
        rng = gr.Plot(label="Operating range over vocab size")

        run_dd.change(refresh, inputs=run_dd, outputs=[md, bars, hm, rng])
        reload_btn.click(lambda: gr.Dropdown(choices=run_names()), outputs=run_dd).then(
            refresh, inputs=run_dd, outputs=[md, bars, hm, rng])
        demo.load(refresh, inputs=run_dd, outputs=[md, bars, hm, rng])

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
