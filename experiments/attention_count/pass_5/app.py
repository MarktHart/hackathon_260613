"""Gradio app for attention_count / pass_5 (trained checkpoint).

Demo tab visualises the artefacts written by main.py:
  1. per-head offset-5 attention score with the 0.5 count line — the count IS
     the number of bars above the line (graded, not saturated to 1.0);
  2. causal ablation of the model's own copy logits — knocking out BOTH counted
     heads collapses the copy; the 6 distractors are dead weight;
  3. operating range — count vs sequence length (16→512) and vs input noise,
     showing where the method holds and where it breaks;
  4. training loss curve + strawman counts (untrained → 0, all-seeded → 8).

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


def _run_paths():
    base = os.path.join(ATTEMPT_DIR, "results")
    return sorted(glob.glob(os.path.join(base, "*", "artifacts.json")))


def _load(path):
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    runs = _run_paths()
    if not runs:
        return None
    with open(runs[-1]) as f:
        return json.load(f)


def _choices():
    return [(os.path.basename(os.path.dirname(p)), p) for p in _run_paths()]


def _empty(msg):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12)
    ax.axis("off")
    return fig


def plot_scores(path):
    art = _load(path)
    if art is None:
        return _empty("No runs yet — execute main.py first.")
    sc = art["per_head_scores"]
    thr = art.get("threshold", 0.5)
    ind = set(art.get("induction_idx", [0, 4]))
    labels = [f"L{i // 4}H{i % 4}" for i in range(len(sc))]
    colors = ["#d62728" if i in ind else "#9aa0a6" for i in range(len(sc))]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(sc)), sc, color=colors)
    ax.axhline(thr, color="black", ls="--", lw=1, label=f"count threshold = {thr}")
    ax.set_xticks(range(len(sc)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("offset-5 attention  attn[last → last-5]")
    ax.set_title(
        f"Trained per-head induction score → count @ {thr} = "
        f"{art.get('predicted_count')}  (ground truth = {art.get('ground_truth', 2)})"
    )
    ax.legend(loc="upper right")
    for i, s in enumerate(sc):
        ax.text(i, s + 0.02, f"{s:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_ablation(path):
    art = _load(path)
    if art is None:
        return _empty("No runs yet — execute main.py first.")
    ca = art["causal_copy_accuracy"]
    order = [
        ("full", "full model"),
        ("ablate_L0H0", "− L0H0"),
        ("ablate_L1H0", "− L1H0"),
        ("ablate_both_induction", "− both counted"),
        ("ablate_all_distractors", "− all 6 distractors"),
    ]
    names = [lbl for k, lbl in order if k in ca]
    vals = [ca[k] for k, _ in order if k in ca]
    colors = ["#2ca02c", "#ff7f0e", "#ff7f0e", "#d62728", "#9aa0a6"][: len(vals)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(vals)), vals, color=colors)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("copy accuracy (model's own logits)")
    ax.set_title("Causal check: removing BOTH counted heads breaks the copy; "
                 "distractors don't matter")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig


def plot_range(path):
    art = _load(path)
    if art is None:
        return _empty("No runs yet — execute main.py first.")
    ls = art.get("length_sweep", [])
    ns = art.get("noise_sweep", [])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    Ls = [d["L"] for d in ls]
    cnt = [d["count"] for d in ls]
    ind = [d["ind_score_mean"] for d in ls]
    dis = [d["distractor_score_mean"] for d in ls]
    ax.plot(Ls, ind, "o-", color="#d62728", label="counted-head score")
    ax.plot(Ls, dis, "s-", color="#9aa0a6", label="distractor score")
    ax.set_xscale("log")
    ax.axhline(0.5, color="black", ls="--", lw=1)
    for x, c in zip(Ls, cnt):
        ax.text(x, 0.92, f"n={c}", ha="center", fontsize=8)
    ax.set_xlabel("sequence length (log)")
    ax.set_ylabel("offset-5 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Operating range: sequence length 16→512")
    ax.legend(loc="center right", fontsize=8)

    ax = axes[1]
    xs = [max(d["noise"], 1e-4) for d in ns]
    cnt = [d["count"] for d in ns]
    indm = [d["ind_score_mean"] for d in ns]
    ax.plot(xs, indm, "o-", color="#d62728", label="counted-head score")
    ax.set_xscale("log")
    ax.axhline(0.5, color="black", ls="--", lw=1)
    for x, c in zip(xs, cnt):
        ax.text(x, 0.05, f"n={c}", ha="center", fontsize=8, rotation=90)
    ax.set_xlabel("input noise std (log)")
    ax.set_ylabel("counted-head offset-5 score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Operating range: input noise 1e-3→1e1")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_training(path):
    art = _load(path)
    if art is None:
        return _empty("No runs yet — execute main.py first.")
    h = art.get("train_history", [])
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if h:
        ax.plot([d["step"] for d in h], [d["loss"] for d in h], color="#1f77b4")
    ax.set_xlabel("training step")
    ax.set_ylabel("copy cross-entropy")
    ax.set_title("Training: the copy/value circuit is learned by gradient descent")
    fig.tight_layout()
    return fig


def summary_md(path):
    art = _load(path)
    if art is None:
        return "No runs yet — execute `main.py` first."
    h = art.get("train_history", [])
    loss = h[-1]["loss"] if h else float("nan")
    return (
        f"**Trained checkpoint.**  Predicted induction heads @0.5: "
        f"**{art.get('predicted_count')}**  (ground truth **{art.get('ground_truth', 2)}**)\n\n"
        f"Final train loss: `{loss:.4f}`  ·  seed-reseed counts: "
        f"{[d['count'] for d in art.get('seed_sweep', [])]}\n\n"
        f"**Strawman — untrained checkpoint:** count = "
        f"{art.get('strawman_untrained_count')} (all heads near-uniform)\n\n"
        f"**Strawman — every head seeded offset-5:** count = "
        f"{art.get('strawman_all_seeded_count')} (over-counts to 8)"
    )


def refresh(path):
    return (plot_scores(path), plot_ablation(path), plot_range(path),
            plot_training(path), summary_md(path))


with gr.Blocks() as demo:
    gr.Markdown(
        "# attention_count · pass_5 — trained checkpoint\n"
        "`base_model.py` minus the MLP, plus a per-head relative-position bias. "
        "Two heads (L0H0, L1H0) are seeded toward offset −5; **everything is then "
        "trained end-to-end** on the copy task. The count below is read off the "
        "trained weights via a real CUDA forward pass."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            ch = _choices()
            run_dd = gr.Dropdown(
                choices=ch,
                value=(ch[-1][1] if ch else None),
                label="Run (defaults to latest)",
            )
            summary = gr.Markdown(summary_md(ch[-1][1] if ch else None))
            scores_plot = gr.Plot(label="Per-head induction score")
            ablation_plot = gr.Plot(label="Causal ablation — copy accuracy")
            range_plot = gr.Plot(label="Operating range — length & noise")
            train_plot = gr.Plot(label="Training loss")

            run_dd.change(refresh, inputs=run_dd,
                          outputs=[scores_plot, ablation_plot, range_plot,
                                   train_plot, summary])
            demo.load(refresh, inputs=run_dd,
                      outputs=[scores_plot, ablation_plot, range_plot,
                               train_plot, summary])

        with gr.Tab("Benchmark"):
            gr.Markdown("## Cross-attempt leaderboard")
            try:
                from agentic.experiments import benchmark_panel
                benchmark_panel(GOAL_DIR)
            except Exception as e:  # keep boot-check alive on any API drift
                gr.Markdown(f"_Benchmark panel unavailable: {e}_")


if __name__ == "__main__":
    demo.launch()
