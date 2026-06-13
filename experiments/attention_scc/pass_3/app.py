"""attention_scc / pass_3 — Gradio app.

Demo tab: the SCC capacity curve for the hand-built beta-softmax head vs the
vanilla 1/sqrt(d) head and the 1/K chance line, plus a temperature sweep, an
ablation bar chart, and the logit-gap structure that explains *why* it works.

Benchmark tab: the shared cross-attempt leaderboard.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel

ATTEMPT_DIR = Path(__file__).parent
GOAL_DIR = ATTEMPT_DIR.parent
RESULTS_ROOT = ATTEMPT_DIR / "results"


# ----------------------------------------------------------------------
# Loading.
# ----------------------------------------------------------------------
def list_runs():
    if not RESULTS_ROOT.exists():
        return []
    runs = [p for p in RESULTS_ROOT.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.name, reverse=True)
    return runs


def run_names():
    return [p.name for p in list_runs()]


def load_diag(run_name):
    if run_name is None:
        runs = list_runs()
        if not runs:
            return None
        run_dir = runs[0]
    else:
        run_dir = RESULTS_ROOT / run_name
    f = run_dir / "diagnostics.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def trapz_norm(xs, ys):
    n = len(xs)
    if n < 2:
        return float(ys[0]) if ys else 0.0
    area = sum(0.5 * (ys[i] + ys[i + 1]) * (xs[i + 1] - xs[i]) for i in range(n - 1))
    width = xs[-1] - xs[0]
    return float(area / width) if width > 0 else float(ys[0])


# ----------------------------------------------------------------------
# Plots.
# ----------------------------------------------------------------------
def plot_scc(d):
    ms = d["method_sweep"]
    bs = d["baseline_standard_sweep"]
    rhos = [r["rho"] for r in ms]
    method = [r["target_attention_mean"] for r in ms]
    method_sd = [r["target_attention_std"] for r in ms]
    base = [r["target_attention_mean"] for r in bs]
    chance = [r["chance_level"] for r in ms]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.errorbar(rhos, method, yerr=method_sd, fmt="o-", capsize=4, lw=2.4,
                ms=8, color="tab:green",
                label=f"hand-built β-softmax (β≈{d['beta_opt']:.0f})")
    ax.plot(rhos, base, "s--", lw=2, ms=7, color="tab:red",
            label="vanilla 1/√d softmax")
    ax.plot(rhos, chance, ":", lw=1.6, color="gray", label="chance = 1/K")
    ax.set_xscale("log", base=2)
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}\n(K={r2['K']})" for r, r2 in zip(rhos, ms)])
    ax.set_xlabel("superposition ratio  ρ = K / d")
    ax.set_ylabel("target attention mass")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Superposition Capacity Curve — temperature is the bottleneck")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", fontsize=9)
    auc = trapz_norm(rhos, method)
    ax.text(0.98, 0.5, f"scc_auc = {auc:.3f}", transform=ax.transAxes,
            ha="right", va="center", fontsize=11,
            bbox=dict(boxstyle="round", fc="honeydew", ec="tab:green"))
    fig.tight_layout()
    return fig


def plot_temperature(d):
    ts = sorted(d["temperature_sweep"], key=lambda r: r["scale"])
    scales = [r["scale"] for r in ts]
    aucs = [r["scc_auc"] for r in ts]

    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.plot(scales, aucs, "o-", lw=2, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("inverse-temperature scale  (logit multiplier)")
    ax.set_ylabel("scc_auc (capacity)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Capacity vs softmax temperature")
    ax.grid(True, alpha=0.3, which="both")

    ax.axvline(d["scale_std"], color="tab:red", ls="--", lw=1.5)
    ax.text(d["scale_std"], 0.55, " vanilla 1/√d\n (fails)", color="tab:red",
            fontsize=9, rotation=90, va="bottom")
    ax.axvline(d["beta_opt"], color="tab:green", ls="--", lw=1.5)
    ax.text(d["beta_opt"], 0.45, " Bayes-optimal β\n (submission)", color="tab:green",
            fontsize=9, rotation=90, va="bottom", ha="right")
    fig.tight_layout()
    return fig


def plot_ablation(d):
    abl = d["ablations"]
    names = [a["name"] for a in abl]
    aucs = [a["scc_auc"] for a in abl]
    colors = ["tab:green", "tab:orange", "tab:red", "gray"]
    chance_auc = abl[0]["chance_auc"]

    fig, ax = plt.subplots(figsize=(7, 4.0))
    bars = ax.bar(range(len(names)), aucs, color=colors[:len(names)])
    ax.axhline(chance_auc, color="black", ls=":", lw=1.4,
               label=f"chance auc = {chance_auc:.3f}")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8.5)
    ax.set_ylabel("scc_auc")
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation — knock out one piece of the circuit")
    for b, a in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.02, f"{a:.3f}",
                ha="center", fontsize=9)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_gap(d):
    lg = d["logit_gap"]
    rhos = [r["rho"] for r in lg]
    gaps = [r["mean_gap"] for r in lg]
    sds = [r["std_gap"] for r in lg]
    frac = [r["frac_target_argmax"] for r in lg]

    fig, ax = plt.subplots(figsize=(7, 4.0))
    ax.errorbar(rhos, gaps, yerr=sds, fmt="o-", capsize=4, lw=2, color="tab:purple",
                label="target − best-distractor logit")
    ax.axhline(0.0, color="black", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(rhos)
    ax.set_xticklabels([f"{r:g}" for r in rhos])
    ax.set_xlabel("superposition ratio  ρ = K / d")
    ax.set_ylabel("logit gap (target − max distractor)")
    ax.set_title("Why it works: the target stays the argmax at every ρ")
    ax.grid(True, alpha=0.3)
    for r, g, fr in zip(rhos, gaps, frac):
        ax.text(r, g + 0.02, f"{fr*100:.0f}% argmax", ha="center", fontsize=8)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def render(run_name):
    d = load_diag(run_name)
    if d is None:
        msg = "No results found. Run `main.py` first to produce diagnostics.json."
        return msg, None, None, None, None
    ms = d["method_sweep"]
    rhos = [r["rho"] for r in ms]
    means = [r["target_attention_mean"] for r in ms]
    auc = trapz_norm(rhos, means)
    cap = next((r["rho"] for r in reversed(ms) if r["target_attention_mean"] >= 0.9), None)
    summary = (
        f"### Hand-built single attention head\n"
        f"- **scc_auc (headline):** {auc:.3f}  (chance ≈ {trapz_norm(rhos, [r['chance_level'] for r in ms]):.3f})\n"
        f"- **Bayes-optimal β:** {d['beta_opt']:.1f}  (vanilla scale 1/√d = {d['scale_std']:.3f})\n"
        f"- **target attention ≥ 0.9 up to ρ =** {cap if cap is not None else '—'} "
        f"(K up to {int(max(rhos) * d['d'])})\n\n"
        f"The head resolves up to 4× overcomplete superposition almost perfectly; "
        f"the *only* difference from the failing vanilla head is the softmax temperature."
    )
    return summary, plot_scc(d), plot_temperature(d), plot_ablation(d), plot_gap(d)


# ----------------------------------------------------------------------
# App.
# ----------------------------------------------------------------------
with gr.Blocks(title="attention_scc / pass_3") as demo:
    gr.Markdown(
        "# Attention SCC — temperature, not geometry, is the capacity bottleneck\n"
        "A single hand-built attention head with a **Bayes-optimal inverse temperature** "
        "(derived from the noise model, not learned) lands ~all attention mass on the target "
        "key up to ρ = K/d = 4. The vanilla 1/√d scaling collapses to chance on the *same* head."
    )
    with gr.Tabs():
        with gr.TabItem("Demo"):
            with gr.Row():
                run_dd = gr.Dropdown(
                    choices=run_names(), value=(run_names()[0] if run_names() else None),
                    label="results run", scale=3,
                )
                refresh = gr.Button("↻ refresh", scale=1)
            summary = gr.Markdown()
            with gr.Row():
                scc_plot = gr.Plot(label="SCC curve")
                temp_plot = gr.Plot(label="temperature sweep")
            with gr.Row():
                abl_plot = gr.Plot(label="ablation")
                gap_plot = gr.Plot(label="logit gap")

            outs = [summary, scc_plot, temp_plot, abl_plot, gap_plot]
            run_dd.change(render, inputs=[run_dd], outputs=outs)

            def _refresh():
                names = run_names()
                val = names[0] if names else None
                return gr.update(choices=names, value=val)

            refresh.click(_refresh, inputs=[], outputs=[run_dd]).then(
                render, inputs=[run_dd], outputs=outs
            )
            demo.load(render, inputs=[run_dd], outputs=outs)

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt leaderboard")
            benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
