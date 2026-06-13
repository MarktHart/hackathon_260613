"""Gradio app for attention_lis / pass_2.

Demo tab: every panel is rendered from artefacts saved by main.py
(results/<run>/viz.npz + extras.json) — no dead code, no recompute drift.
Benchmark tab: shared cross-attempt panel.
"""
import json
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS = ATTEMPT_DIR / "results"


def run_choices():
    if not RESULTS.exists():
        return []
    return sorted([d.name for d in RESULTS.iterdir() if d.is_dir()], reverse=True)


def load_run(run_name):
    if not run_name:
        return None
    run_dir = RESULTS / run_name
    viz_p = run_dir / "viz.npz"
    extras_p = run_dir / "extras.json"
    if not viz_p.exists() or not extras_p.exists():
        return None
    viz = np.load(viz_p)
    with open(extras_p) as f:
        extras = json.load(f)
    return {
        "q_proj": viz["q_proj"],
        "factors": viz["factors"],
        "cos_q": viz["cos_q"],
        "sweep_noise": viz["sweep_noise"],
        "sweep_ortho": viz["sweep_ortho"],
        "sweep_align": viz["sweep_align"],
        "extras": extras,
    }


def fig_compare(extras):
    labels = ["trained\nattn (q)", "untrained\n(strawman)", "hand-built\n(ideal)", "linear\nbaseline"]
    vals = [extras["ortho_trained"], extras["ortho_untrained"],
            extras["ortho_handbuilt"], extras["ortho_baseline"]]
    colors = ["#2b8cbe", "#d6604d", "#4daf4a", "#999999"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("LIS orthogonality  (1 − |cos|)")
    ax.set_title("Subspace independence of the query encoding")
    ax.axhline(extras["ortho_baseline"], ls="--", color="#999999", lw=1)
    fig.tight_layout()
    return fig


def fig_faithfulness(extras):
    labels = ["trained\nattention", "attention\nablated (uniform)"]
    vals = [extras["recon_acc_trained"], extras["recon_acc_ablated"]]
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    bars = ax.bar(labels, vals, color=["#2b8cbe", "#d6604d"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, ls="--", color="#999999", lw=1, label="chance")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("factor reconstruction accuracy")
    ax.set_title("Faithfulness: knock out the attention circuit")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_cos(cos_q):
    K = cos_q.shape[0]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(cos_q, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"f{i}" for i in range(K)])
    ax.set_yticklabels([f"f{i}" for i in range(K)])
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cos_q[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(cos_q[i, j]) > 0.5 else "black", fontsize=8)
    ax.set_title("Query encoding-direction cosines\n(off-diagonal ≈ 0 ⇒ orthogonal)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_sweep(noise, ortho, align):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(noise, ortho, "o-", color="#2b8cbe", label="orthogonality")
    ax.plot(noise, align, "s-", color="#e08214", label="alignment")
    ax.set_xlabel("input noise_std")
    ax.set_ylabel("metric")
    ax.set_ylim(0, 1.08)
    ax.set_title("Operating range: robustness to input noise")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_separation(q_proj, factors):
    K = q_proj.shape[0]
    fig, axes = plt.subplots(1, K, figsize=(3.0 * K, 3.0), squeeze=False)
    for k in range(K):
        ax = axes[0, k]
        pk = q_proj[k]
        fk = factors[:, k]
        ax.hist(pk[fk > 0], bins=15, alpha=0.6, color="#2b8cbe", density=True, label="+1")
        ax.hist(pk[fk < 0], bins=15, alpha=0.6, color="#d6604d", density=True, label="−1")
        ax.set_title(f"factor {k}")
        ax.set_xlabel("q · factor_dir")
        if k == 0:
            ax.set_ylabel("density")
            ax.legend(fontsize=7)
    fig.suptitle("Each query axis separates its factor (±1)", y=1.02)
    fig.tight_layout()
    return fig


def render(run_name):
    data = load_run(run_name)
    if data is None:
        empty = plt.figure()
        return ("### No run found — execute `main.py` first.",
                empty, empty, empty, empty, empty)
    e = data["extras"]
    md = (
        f"### Run `{run_name}`\n"
        f"- **orthogonality (trained q):** {e['ortho_trained']:.3f}  "
        f"(linear baseline {e['ortho_baseline']:.3f}, "
        f"lift **+{e['lift_over_baseline']:.3f}**)\n"
        f"- **alignment (trained q):** {e['align_trained']:.3f}\n"
        f"- **robustness:** {e['robustness']:.3f}\n"
        f"- **factor recon accuracy:** {e['recon_acc_trained']:.3f} with attention "
        f"vs {e['recon_acc_ablated']:.3f} ablated\n"
    )
    return (
        md,
        fig_compare(e),
        fig_faithfulness(e),
        fig_cos(data["cos_q"]),
        fig_sweep(data["sweep_noise"], data["sweep_ortho"], data["sweep_align"]),
        fig_separation(data["q_proj"], data["factors"]),
    )


def _default_run():
    ch = run_choices()
    return ch[0] if ch else None


with gr.Blocks() as demo:
    gr.Markdown("# attention_lis — pass_2 (trained single attention layer)")
    gr.Markdown(
        "A one-layer self-attention block trained on a **diagonal-masked factor "
        "retrieval** task. Positions must attend to *others* sharing their factor "
        "combo, forcing queries to encode the K factors in linearly-independent "
        "subspaces."
    )

    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(choices=run_choices(), value=_default_run(),
                                     label="run", interactive=True, scale=3)
                reload_btn = gr.Button("Reload", scale=1)
            info = gr.Markdown()
            with gr.Row():
                p_compare = gr.Plot(label="Orthogonality vs strawman / ideal")
                p_faith = gr.Plot(label="Faithfulness ablation")
            with gr.Row():
                p_cos = gr.Plot(label="Encoding-direction cosines")
                p_sweep = gr.Plot(label="Noise robustness")
            p_sep = gr.Plot(label="Factor separation")

            outs = [info, p_compare, p_faith, p_cos, p_sweep, p_sep]
            run_dd.change(render, inputs=run_dd, outputs=outs)
            reload_btn.click(render, inputs=run_dd, outputs=outs)
            demo.load(render, inputs=run_dd, outputs=outs)

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt benchmark history")
            benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
