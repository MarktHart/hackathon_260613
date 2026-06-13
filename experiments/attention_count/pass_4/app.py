"""Gradio app for attention_count / pass_4.

Demo tab visualises the REAL measured artefacts written by main.py:
  1. per-head induction scores (attn[63,58]) with the 0.5 count threshold —
     the bar that, if flipped, would change the predicted count;
  2. the causal ablation: copy accuracy of the model's own logits when each
     head is knocked out — the evidence that the counted heads are the ones
     the model actually uses;
  3. strawman counts (no-induction / all-eight) under the same measurement.

Benchmark tab drops in the shared cross-attempt panel.
"""
import glob
import json
import os

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)


def _run_dirs() -> list[str]:
    base = os.path.join(ATTEMPT_DIR, "results")
    return sorted(glob.glob(os.path.join(base, "*", "artifacts.json")))


def _load(run_path: str | None):
    if run_path and os.path.isfile(run_path):
        with open(run_path) as f:
            return json.load(f)
    runs = _run_dirs()
    if not runs:
        return None
    with open(runs[-1]) as f:
        return json.load(f)


def _dropdown_choices():
    runs = _run_dirs()
    return [(os.path.basename(os.path.dirname(p)), p) for p in runs]


def _empty_fig(msg: str):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return fig


def plot_scores(run_path):
    art = _load(run_path)
    if art is None:
        return _empty_fig("No runs yet — execute main.py first.")
    scores = art["per_head_scores"]
    thr = art.get("threshold", 0.5)
    ind = set(art.get("induction_heads_layer_major_idx", [0, 4]))
    labels = [f"L{i // 4}H{i % 4}" for i in range(len(scores))]
    colors = ["#d62728" if i in ind else "#9aa0a6" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(scores)), scores, color=colors)
    ax.axhline(thr, color="black", ls="--", lw=1, label=f"count threshold = {thr}")
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("induction score  =  attn[target=63 → source=58]")
    ax.set_title(
        f"Per-head induction score  →  predicted count @ {thr} = "
        f"{art.get('predicted_count_thr0p5')}  (ground truth = 2)"
    )
    ax.legend(loc="upper right")
    for i, s in enumerate(scores):
        ax.text(i, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ablation(run_path):
    art = _load(run_path)
    if art is None:
        return _empty_fig("No runs yet — execute main.py first.")
    ca = art["causal_copy_accuracy"]
    order = [
        ("full", "full model"),
        ("ablate_layer0_head0", "− L0H0 (induction)"),
        ("ablate_layer1_head0", "− L1H0 (induction)"),
        ("ablate_both_induction", "− both induction"),
        ("ablate_all_distractors", "− all 6 distractors"),
    ]
    names = [lbl for k, lbl in order if k in ca]
    vals = [ca[k] for k, _ in order if k in ca]
    colors = ["#2ca02c", "#d62728", "#d62728", "#d62728", "#9aa0a6"][: len(vals)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("copy-task accuracy (model's own logits)")
    ax.set_title("Causal check: knocking out an induction head breaks the copy; "
                 "distractors don't")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def summary_md(run_path):
    art = _load(run_path)
    if art is None:
        return "No runs yet — execute `main.py` first."
    return (
        f"**Predicted induction heads @0.5:** {art.get('predicted_count_thr0p5')}  "
        f"(ground truth **2**)\n\n"
        f"**Strawman — no induction heads:** count = "
        f"{art.get('strawman_uniform_count')} (all heads attend uniformly)\n\n"
        f"**Strawman — all 8 wired as induction:** count = "
        f"{art.get('strawman_alleight_count')} (over-counts)\n\n"
        f"τ={art.get('tau')}, copy gain α={art.get('alpha')}, delay={art.get('delay')}"
    )


def refresh(run_path):
    return plot_scores(run_path), plot_ablation(run_path), summary_md(run_path)


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_count · pass_4 — hand-built induction circuit (real GPU forward)\n"
        "Two heads (L0H0, L1H0) are hand-wired to attend with offset −5; the other "
        "six have zero QK weights and attend uniformly. Everything below is read off "
        "a **real** causal-softmax forward pass."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            choices = _dropdown_choices()
            run_dd = gr.Dropdown(
                choices=choices,
                value=(choices[-1][1] if choices else None),
                label="Run (defaults to latest)",
            )
            summary = gr.Markdown(summary_md(choices[-1][1] if choices else None))
            scores_plot = gr.Plot(label="Per-head induction scores")
            ablation_plot = gr.Plot(label="Causal ablation — copy accuracy")

            run_dd.change(refresh, inputs=run_dd,
                          outputs=[scores_plot, ablation_plot, summary])
            demo.load(refresh, inputs=run_dd,
                      outputs=[scores_plot, ablation_plot, summary])

        with gr.Tab("Benchmark"):
            gr.Markdown("## Cross-attempt leaderboard")
            try:
                gr.Markdown(GOAL_DIR)  # benchmark_panel renders below
                from agentic.experiments import benchmark_panel
                benchmark_panel(GOAL_DIR)
            except Exception as e:  # keep boot-check alive on any API drift
                gr.Markdown(f"_Benchmark panel unavailable: {e}_")


if __name__ == "__main__":
    demo.launch()
