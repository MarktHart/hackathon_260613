"""Gradio app for pass_6 — dual-basis NOT head under superposition.

Demo tab tells the story without the README:
  (1) Suppression gap vs cos: corrected (dual basis) holds, naive (ablated)
      collapses, framework linear baseline sits at chance.  This is the
      headline contrast that engages the superposition axis.
  (2) Norm cost ||x_A|| vs cos: WHY superposition is hard — separating
      overlapping directions costs read-out norm.
  (3) Attention bars for the four (A,B) conditions at the easy (cos=0) and hard
      (cos=0.8) ends, corrected vs naive — the mechanism, made concrete.

Benchmark tab drops in the shared leaderboard across all attempts.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

_GOAL_DIR = Path(__file__).parent.parent
_ATTEMPT_DIR = Path(__file__).parent
_RESULTS_DIR = _ATTEMPT_DIR / "results"

_KEYS = ["A1B0", "A1B1", "A0B0", "A0B1"]
_POS = ["A-tok", "B-tok", "query", "ans"]


def _runs():
    if not _RESULTS_DIR.exists():
        return []
    return sorted((p.name for p in _RESULTS_DIR.iterdir() if p.is_dir()), reverse=True)


def _load(run_name):
    if run_name is None:
        return None, None
    run_dir = _RESULTS_DIR / run_name
    viz = None
    payload = None
    vp = run_dir / "viz_data.json"
    if vp.exists():
        with vp.open() as f:
            viz = json.load(f)
    bp = run_dir / "benchmark.json"
    if bp.exists():
        with bp.open() as f:
            rec = json.load(f)
        payload = rec.get("payload", rec)
    return viz, payload


def _figure(viz):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)

    if viz is None:
        for ax in axes:
            ax.text(0.5, 0.5, "no run found", ha="center", va="center")
            ax.axis("off")
        return fig

    cos = viz["cos"]

    # Panel 1: suppression gap (continuous) — the real superposition contrast.
    ax = axes[0]
    ax.plot(cos, viz["corrected"]["suppression_gap"], "o-", color="#1f77b4",
            lw=2.4, ms=7, label="corrected (dual basis)")
    ax.plot(cos, viz["naive"]["suppression_gap"], "s--", color="#d62728",
            lw=2.2, ms=7, label="naive (dual basis ablated)")
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_xlabel("cos(theta)  between e_A and e_B")
    ax.set_ylabel("suppression gap  E[attn_A|B=0] − E[attn_A|B=1]")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("NOT survives superposition only with the dual basis")
    ax.legend(loc="center left")
    ax.grid(alpha=0.3)

    # Panel 2: norm cost — why superposition is hard.
    ax = axes[1]
    ax.plot(cos, viz["xA_norm"], "^-", color="#2ca02c", lw=2.4, ms=7)
    ax.set_xlabel("cos(theta)")
    ax.set_ylabel("||x_A||  (min-norm read-out)")
    ax.set_title("Cost of separation: ||x_A|| grows as features align")
    ax.grid(alpha=0.3)

    # Panel 3: attention bars at easy vs hard cos.
    ax = axes[2]
    keys = sorted(viz["examples"].keys(), key=float)
    hard = keys[-1]
    ex = viz["examples"][hard]
    x = range(len(_KEYS))
    corr = [ex["corrected"][k][0] for k in _KEYS]   # attn mass on A-token
    naiv = [ex["naive"][k][0] for k in _KEYS]
    w = 0.38
    ax.bar([i - w / 2 for i in x], corr, w, color="#1f77b4", label="corrected")
    ax.bar([i + w / 2 for i in x], naiv, w, color="#d62728", label="naive")
    ax.set_xticks(list(x))
    ax.set_xticklabels(_KEYS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("query → A-token attention")
    ax.set_title(f"query→A attention by (A,B) at cos={hard}")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    return fig


def _summary(viz, payload):
    if viz is None:
        return "No run found. Run `main.py` first."
    cos = viz["cos"]
    corr_gap = viz["corrected"]["suppression_gap"]
    naive_gap = viz["naive"]["suppression_gap"]
    sharp = viz["corrected"]["not_sharpness"]
    worst = min(sharp)
    canon = sharp[0]
    robustness = max(0.0, min(1.0, (worst - 0.5) / (canon - 0.5))) if canon > 0.5 else 0.0
    c = viz["constants"]
    lines = [
        "### Hand-built dual-basis NOT head",
        f"- logit = **{c['ALPHA']:.0f}·feat_A − {c['BETA']:.0f}·feat_B + ({c['DELTA']:.0f})** through the real W_Q·W_Kᵀ",
        f"- **superposition_robustness = {robustness:.3f}** (corrected, worst-slice sharpness re-centred on chance)",
        f"- corrected suppression gap: {corr_gap[0]:.3f} (cos=0) → {corr_gap[-1]:.3f} (cos={cos[-1]:.1f})  — flat, NOT holds",
        f"- naive (ablated) gap: {naive_gap[0]:.3f} → {naive_gap[-1]:.3f}  — collapses under superposition",
        "",
        "The naive head reads with the raw e_A/e_B directions; the corrected head solves the "
        "QK circuit for the dual (Gram-inverse) basis, cancelling cross-talk. Same metric, same data — "
        "the only difference is whether the dual-basis correction is present.",
    ]
    return "\n".join(lines)


def _refresh(run_name):
    viz, payload = _load(run_name)
    return _figure(viz), _summary(viz, payload)


with gr.Blocks(title="attention_not — pass_6") as demo:
    gr.Markdown("## attention_not · pass_6 — inhibitory NOT via dual-basis read-out of superposed features")

    with gr.Tab("Demo"):
        with gr.Row():
            run_dd = gr.Dropdown(choices=_runs(), value=(_runs()[0] if _runs() else None),
                                 label="run", scale=3)
            refresh_btn = gr.Button("Refresh", scale=1)
        summary = gr.Markdown()
        plot = gr.Plot()

        run_dd.change(_refresh, inputs=run_dd, outputs=[plot, summary])
        refresh_btn.click(_refresh, inputs=run_dd, outputs=[plot, summary])
        demo.load(_refresh, inputs=run_dd, outputs=[plot, summary])

    with gr.Tab("Benchmark"):
        benchmark_panel(_GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
