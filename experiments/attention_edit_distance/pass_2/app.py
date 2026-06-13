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


def get_runs():
    if not RESULTS_DIR.exists():
        return []
    runs = [d.name for d in RESULTS_DIR.iterdir()
            if d.is_dir() and (d / "benchmark.json").exists()]
    runs.sort(reverse=True)
    return runs


def _load(run_name, fname):
    p = RESULTS_DIR / run_name / fname
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def _ablation_plot(run_name):
    abl = _load(run_name, "ablation.json")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if abl is None:
        ax.text(0.5, 0.5, "no ablation.json", ha="center", va="center")
        return fig
    e = abl["edit_distance"]
    ax.errorbar(e, abl["full_mean"], yerr=abl["full_std"], fmt="o-",
                color="#1f77b4", capsize=3, lw=2, ms=7,
                label=f"Full content head  (ρ={abl['spearman_full']:.2f})")
    ax.errorbar(e, abl["ablated_mean"], yerr=abl["ablated_std"], fmt="s--",
                color="#d62728", capsize=3, lw=1.8, ms=6, alpha=0.85,
                label=f"Content ABLATED  (ρ={abl['spearman_ablated']:.2f})")
    ax.errorbar(e, abl["baseline_mean"], yerr=abl["baseline_std"], fmt="^:",
                color="#7f7f7f", capsize=3, lw=1.5, ms=6, alpha=0.7,
                label=f"Random-attn baseline (ρ={abl['spearman_baseline']:.2f})")
    ax.set_xlabel("Levenshtein edit distance", fontsize=12)
    ax.set_ylabel("Attention distance (1 − cosine)", fontsize=12)
    ax.set_title("Hand-built content head vs causal ablation\n"
                 "knocking out content collapses the monotonic curve", fontsize=12)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def _oprange_plot(run_name):
    op = _load(run_name, "operating_range.json")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if not op:
        ax1.text(0.5, 0.5, "no operating_range.json", ha="center", va="center")
        return fig
    # left: curves per regime
    cmap = plt.cm.viridis
    for i, cfg in enumerate(op):
        c = cmap(i / max(1, len(op) - 1))
        ax1.plot(cfg["edit"], cfg["mean"], "o-", color=c, ms=4,
                 label=f"{cfg['label']}")
    ax1.set_xlabel("edit distance")
    ax1.set_ylabel("attention distance (1 − cosine)")
    ax1.set_title("Curve holds across regimes")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    # right: spearman per regime
    labels = [c["label"] for c in op]
    rhos = [c["spearman"] for c in op]
    colors = [cmap(i / max(1, len(op) - 1)) for i in range(len(op))]
    ax2.barh(range(len(op)), rhos, color=colors)
    ax2.set_yticks(range(len(op)))
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlim(0, 1.05)
    ax2.axvline(1.0, color="k", ls=":", lw=1)
    ax2.set_xlabel("Spearman ρ (edit dist vs attn dist)")
    ax2.set_title("Monotonicity across ~1.7 orders of vocab")
    for i, r in enumerate(rhos):
        ax2.text(min(r + 0.02, 0.98), i, f"{r:.2f}", va="center", fontsize=8)
    ax2.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    return fig


def _summary(run_name):
    abl = _load(run_name, "ablation.json")
    bm = _load(run_name, "benchmark.json")
    if abl is None:
        return f"**Run:** {run_name}\n\n(no ablation data)"
    lines = [
        f"**Run:** `{run_name}`",
        "",
        "**Mechanism:** hand-built, frozen single causal attention head "
        "(`base_model.py` embedding + 1 head, MLP dropped, random weights).",
        "",
        f"- Full content head — Spearman ρ = **{abl['spearman_full']:.3f}**",
        f"- Content ABLATED (position-only) — ρ = **{abl['spearman_ablated']:.3f}**  "
        "→ attention is token-independent, distance ≈ 0, correlation gone.",
        f"- Random-attention baseline — ρ = **{abl['spearman_baseline']:.3f}**",
    ]
    if bm and "metrics" in bm:
        m = bm["metrics"]
        if "lift_over_baseline" in m:
            lines.append(f"- Lift over baseline = **{m['lift_over_baseline']:.3f}**")
    return "\n".join(lines)


def refresh(run_name):
    if not run_name:
        return None, None, "No runs found. Run main.py first."
    return _ablation_plot(run_name), _oprange_plot(run_name), _summary(run_name)


with gr.Blocks(title="Attention Edit Distance — pass_2") as demo:
    gr.Markdown(
        "# Attention Edit Distance — pass_2\n"
        "A **hand-built** single causal attention head (frozen random weights, a "
        "minimal delta from `base_model.py`). Does attention-pattern distance grow "
        "monotonically with Levenshtein edit distance — and is the *content* pathway "
        "causally responsible?"
    )
    with gr.Tab("Demo"):
        runs = get_runs()
        with gr.Row():
            run_dd = gr.Dropdown(choices=runs, value=runs[0] if runs else None,
                                 label="Run", scale=2)
            refresh_btn = gr.Button("Refresh", scale=1)
        summary_md = gr.Markdown()
        gr.Markdown("### Causal ablation: full head vs content-knockout vs random baseline")
        abl_plot = gr.Plot()
        gr.Markdown("### Operating range: vocab 20→1000, seq 8→128")
        op_plot = gr.Plot()

        run_dd.change(refresh, inputs=run_dd, outputs=[abl_plot, op_plot, summary_md])
        refresh_btn.click(refresh, inputs=run_dd, outputs=[abl_plot, op_plot, summary_md])
        demo.load(refresh, inputs=run_dd, outputs=[abl_plot, op_plot, summary_md])

    with gr.Tab("Benchmark"):
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
