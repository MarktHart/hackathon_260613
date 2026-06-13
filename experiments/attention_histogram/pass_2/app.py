"""
Gradio app for attention_histogram / pass_2.

Demo tab: the attention histogram of the iterative-refinement head vs. its own
no-refinement ablation vs. the plain dot-product baseline, for a chosen
distractor-similarity slice; plus the sweep of sharpness AND target hit-rate
across the interference axis with the ablation curve overlaid (the causal
evidence that the refinement step is what lifts targeting).

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


def render(run, sim_choice):
    if not run:
        fig = plt.figure(figsize=(6, 3))
        plt.text(0.5, 0.5, "No runs yet — run main.py", ha="center")
        return fig, fig, "No runs found."

    payload, ablation, ex = _load(run)
    sim = float(sim_choice)
    n = ex["n_positions"]

    rec = min(ex["examples"], key=lambda e: abs(e["similarity"] - sim))
    tgt = rec["target_index"]
    xs = list(range(n))

    fig1, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    panels = (
        ("mech_attn", "Iterative refinement (mechanism)"),
        ("ablate_attn", "No refinement (ablation)"),
        ("base_attn", "Plain dot-product (baseline)"),
    )
    for ax, (key, title) in zip(axes, panels):
        attn = rec[key]
        colors = ["#d62728" if i == tgt else "#1f77b4" for i in xs]
        ax.bar(xs, attn, color=colors)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("key position")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("attention weight")
    fig1.suptitle(f"Attention histogram @ sim={rec['similarity']:.1f}  "
                  f"(red = correct key #{tgt})", fontsize=11)
    fig1.tight_layout()

    sims = [s["similarity"] for s in payload["sweep"]]
    m_sharp = [s["attention_sharpness"] for s in payload["sweep"]]
    a_sharp = [s["attention_sharpness"] for s in ablation["sweep"]]
    b_sharp = [s["attention_sharpness"] for s in payload["linear_baseline"]]
    m_hit = [s["target_hit_rate"] for s in payload["sweep"]]
    a_hit = [s["target_hit_rate"] for s in ablation["sweep"]]
    b_hit = [s["target_hit_rate"] for s in payload["linear_baseline"]]

    fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11, 3.4))
    axa.plot(sims, m_sharp, "o-", color="#d62728", label="refinement")
    axa.plot(sims, a_sharp, "^--", color="#ff7f0e", label="no-refine (ablation)")
    axa.plot(sims, b_sharp, "s:", color="#1f77b4", label="dot-product")
    axa.set_title("Histogram sharpness", fontsize=10)
    axa.set_xlabel("distractor↔target cosine")
    axa.set_ylabel("1 − H/log n")
    axa.set_ylim(0, 1.02)
    axa.legend(fontsize=8)

    axb.plot(sims, m_hit, "o-", color="#d62728", label="refinement")
    axb.plot(sims, a_hit, "^--", color="#ff7f0e", label="no-refine (ablation)")
    axb.plot(sims, b_hit, "s:", color="#1f77b4", label="dot-product")
    axb.axhline(payload["chance_hit_rate"], color="gray", ls=":", label="chance")
    axb.set_title("Target hit rate (targeting accuracy)", fontsize=10)
    axb.set_xlabel("distractor↔target cosine")
    axb.set_ylim(0, 1.02)
    axb.legend(fontsize=8)
    fig2.tight_layout()

    canon = payload["canonical_similarity"]
    info = (f"**{payload['model_name']}**  \n"
            f"d={payload['d']}, n_positions={payload['n_positions']}, "
            f"chance={payload['chance_hit_rate']:.3f}  \n"
            f"canonical sim={canon}: hit={m_hit[0]:.3f} "
            f"(no-refine {a_hit[0]:.3f}, base {b_hit[0]:.3f}); "
            f"sharp={m_sharp[0]:.3f} (base {b_sharp[0]:.3f})")
    return fig1, fig2, info


with gr.Blocks() as demo:
    gr.Markdown("# attention_histogram — iterative-refinement attention head")
    gr.Markdown(
        "The head denoises the noisy query against the key set before scoring. "
        "Compare the three histograms and watch the ablation (no-refine) curve "
        "fall back toward the dot-product baseline on targeting.")
    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(
                    choices=list_runs(),
                    value=(list_runs()[0] if list_runs() else None),
                    label="run")
                sim_dd = gr.Dropdown(
                    choices=[str(s) for s in SWEEP],
                    value="0.6", label="distractor similarity")
            info_md = gr.Markdown()
            hist_plot = gr.Plot(label="attention histogram")
            sweep_plot = gr.Plot(label="sweep across interference")

            run_dd.change(render, [run_dd, sim_dd],
                          [hist_plot, sweep_plot, info_md])
            sim_dd.change(render, [run_dd, sim_dd],
                          [hist_plot, sweep_plot, info_md])
            demo.load(render, [run_dd, sim_dd],
                      [hist_plot, sweep_plot, info_md])
        with gr.Tab("Benchmark"):
            benchmark_panel(GOAL_DIR)

if __name__ == "__main__":
    demo.launch()
