"""
Gradio app for attention_histogram / pass_3.

Demo tab:
  1. Three side-by-side attention HISTOGRAMS at a chosen distractor-similarity
     slice: trained denoising head vs. its alpha-knockout ablation
     (temperature only) vs. plain dot-product baseline. Red bar = correct key.
  2. The SWEEP across rising distractor↔target cosine: sharpness AND target
     hit-rate, with the ablation curve overlaid — the causal evidence that the
     denoising block (not the temperature) is what keeps the aim correct.
  3. DEPTH curve: mean hit-rate vs number of denoising blocks (operating
     range), plus the learned scalars (alpha, beta1, beta2) and loss history.

Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""

import json
import os

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = os.path.dirname(os.path.abspath(__file__))
GOAL_DIR = os.path.dirname(ATTEMPT_DIR)
RESULTS_DIR = os.path.join(ATTEMPT_DIR, "results")

SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]


def list_runs():
    if not os.path.isdir(RESULTS_DIR):
        return []
    runs = [d for d in os.listdir(RESULTS_DIR)
            if os.path.isfile(os.path.join(RESULTS_DIR, d, "payload.json"))]
    return sorted(runs, reverse=True)


def _load(run):
    base = os.path.join(RESULTS_DIR, run)
    with open(os.path.join(base, "payload.json")) as f:
        payload = json.load(f)
    with open(os.path.join(base, "ablation.json")) as f:
        ablation = json.load(f)
    with open(os.path.join(base, "examples.json")) as f:
        examples = json.load(f)
    return payload, ablation, examples


def _empty(msg):
    fig = plt.figure(figsize=(6, 3))
    plt.text(0.5, 0.5, msg, ha="center", va="center")
    plt.axis("off")
    return fig


def render(run, sim_choice):
    if not run:
        e = _empty("No runs yet — run main.py")
        return e, e, e, "No runs found."

    payload, ablation, ex = _load(run)
    sim = float(sim_choice)
    n = ex["n_positions"]

    rec = min(ex["examples"], key=lambda e: abs(e["similarity"] - sim))
    tgt = rec["target_index"]
    xs = list(range(n))

    # --- (1) histograms ---
    fig1, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    panels = (
        ("mech_attn", "Trained denoising head (mechanism)"),
        ("ablate_attn", "alpha=0 knockout (temperature only)"),
        ("base_attn", "Plain dot-product (baseline)"),
    )
    for ax, (key, title) in zip(axes, panels):
        colors = ["#d62728" if i == tgt else "#1f77b4" for i in xs]
        ax.bar(xs, rec[key], color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("key position")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("attention weight")
    fig1.suptitle(f"Attention histogram @ sim={rec['similarity']:.1f}  "
                  f"(red = correct key #{tgt})", fontsize=11)
    fig1.tight_layout()

    # --- (2) sweep: sharpness + hit-rate ---
    sims = [s["similarity"] for s in payload["sweep"]]
    m_sharp = [s["attention_sharpness"] for s in payload["sweep"]]
    a_sharp = [s["attention_sharpness"] for s in ablation["sweep"]]
    b_sharp = [s["attention_sharpness"] for s in payload["linear_baseline"]]
    m_hit = [s["target_hit_rate"] for s in payload["sweep"]]
    a_hit = [s["target_hit_rate"] for s in ablation["sweep"]]
    b_hit = [s["target_hit_rate"] for s in payload["linear_baseline"]]

    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 3.4))
    axa.plot(sims, m_sharp, "o-", color="#d62728", label="denoising (mech)")
    axa.plot(sims, a_sharp, "^--", color="#ff7f0e", label="alpha=0 ablation")
    axa.plot(sims, b_sharp, "s:", color="#1f77b4", label="dot-product")
    axa.set_title("Histogram sharpness  (1 − H/log n)", fontsize=10)
    axa.set_xlabel("distractor↔target cosine")
    axa.set_ylabel("sharpness")
    axa.set_ylim(0, 1.02)
    axa.legend(fontsize=8)

    axb.plot(sims, m_hit, "o-", color="#d62728", label="denoising (mech)")
    axb.plot(sims, a_hit, "^--", color="#ff7f0e", label="alpha=0 ablation")
    axb.plot(sims, b_hit, "s:", color="#1f77b4", label="dot-product")
    axb.axhline(payload["chance_hit_rate"], color="gray", ls=":", label="chance")
    axb.set_title("Target hit rate  (aim)", fontsize=10)
    axb.set_xlabel("distractor↔target cosine")
    axb.set_ylim(0, 1.02)
    axb.legend(fontsize=8)
    fig2.tight_layout()

    # --- (3) depth + loss ---
    depth = ex.get("depth", [])
    params = ex.get("params", {})
    fig3, (axd, axl) = plt.subplots(1, 2, figsize=(11, 3.2))
    if depth:
        nis = [d["n_iter"] for d in depth]
        axd.plot(nis, [d["mean_hit"] for d in depth], "o-", color="#2ca02c",
                 label="mean hit rate")
        axd.plot(nis, [d["canonical_sharpness"] for d in depth], "s--",
                 color="#9467bd", label="canonical sharpness")
        axd.set_title("Operating range: depth (denoising blocks)", fontsize=10)
        axd.set_xlabel("n_iter (block 1 repeats)")
        axd.set_xticks(nis)
        axd.set_ylim(0, 1.02)
        axd.legend(fontsize=8)
    hist = params.get("loss_history", [])
    if hist:
        axl.plot(range(len(hist)), hist, "o-", color="#1f77b4")
        axl.set_title("Training loss (cross-entropy)", fontsize=10)
        axl.set_xlabel("checkpoint")
        axl.set_ylabel("loss")
    fig3.tight_layout()

    canon = payload["canonical_similarity"]
    info = (
        f"**{payload['model_name']}**  \n"
        f"learned: beta1={params.get('beta1', float('nan')):.2f}, "
        f"beta2={params.get('beta2', float('nan')):.2f}, "
        f"gamma={params.get('gamma', float('nan')):.2f}, "
        f"gate_w={params.get('gate_w', float('nan')):.2f}, "
        f"gate_b={params.get('gate_b', float('nan')):.2f}  \n"
        f"d={payload['d']}, n_positions={payload['n_positions']}, "
        f"chance={payload['chance_hit_rate']:.3f}  \n"
        f"canonical sim={canon}: hit mech={m_hit[0]:.3f} "
        f"(alpha=0 ablation {a_hit[0]:.3f}, dot-product {b_hit[0]:.3f}); "
        f"sharp mech={m_sharp[0]:.3f} (dot-product {b_sharp[0]:.3f})")
    return fig1, fig2, fig3, info


with gr.Blocks() as demo:
    gr.Markdown("# attention_histogram — trained denoising attention head")
    gr.Markdown(
        "A 2-block attention circuit (denoise the noisy query against the key "
        "set, then score) whose strength `alpha` was **discovered by training**. "
        "The `alpha=0` knockout stays sharp (it keeps the trained temperature) "
        "but its aim collapses back onto the dot-product baseline — so the gap "
        "is exactly the denoising block's causal contribution.")
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(
                    choices=list_runs(),
                    value=(list_runs()[0] if list_runs() else None),
                    label="run")
                sim_dd = gr.Dropdown(
                    choices=[str(s) for s in SWEEP],
                    value="0.8", label="distractor similarity")
            info_md = gr.Markdown()
            hist_plot = gr.Plot(label="attention histogram")
            sweep_plot = gr.Plot(label="sweep across interference")
            depth_plot = gr.Plot(label="depth + training")

            run_dd.change(render, [run_dd, sim_dd],
                          [hist_plot, sweep_plot, depth_plot, info_md])
            sim_dd.change(render, [run_dd, sim_dd],
                          [hist_plot, sweep_plot, depth_plot, info_md])
            demo.load(render, [run_dd, sim_dd],
                      [hist_plot, sweep_plot, depth_plot, info_md])
        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
