"""Gradio app for attention_regex / pass_2.

Demo tab:
  (A) Per-position attention vs ground-truth match-ends for a chosen length+seed.
  (B) Sweep-summary panel: match_sharpness / FPR / FNR across the L=1..6 sweep,
      with the no-composition linear baseline overlaid — the headline
      `length_robustness` made visual.
Benchmark tab: cross-attempt leaderboard via benchmark_panel.
"""

import json
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from agentic.experiments import load_task, benchmark_panel

DEVICE = "cuda"
POS_BIAS = 30.0
BETA = 2.0

HERE = Path(__file__).parent
GOAL_DIR = HERE.parent
TASK = load_task(__file__)
BATCH = TASK.generate(seed=TASK.EVAL_SEED)


# --- the circuit (identical to main.py.model_fn) -------------------------
def model_fn(pattern, embed, residual):
    pt = torch.as_tensor(pattern, dtype=torch.long, device=DEVICE)
    et = torch.as_tensor(embed, dtype=torch.float32, device=DEVICE)
    rt = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    N, d = rt.shape
    L = int(pt.shape[0])
    conc = torch.where(pt >= 0)[0]
    if conc.numel() == 0:
        return torch.zeros(N, dtype=torch.float32, device=DEVICE).detach().cpu().numpy()
    pos = torch.arange(N, device=DEVICE)
    rel = pos[:, None] - pos[None, :]
    hs = []
    for j in conc.tolist():
        shift = L - 1 - j
        sim = rt @ et[pt[j]]
        bias = -POS_BIAS * (rel.float() - float(shift)).abs()
        hs.append(torch.softmax(bias, dim=1) @ sim)
    logit = torch.stack(hs, dim=0).min(dim=0).values
    logit = logit.masked_fill(pos < (L - 1), -1e9)
    return (BETA * logit).detach().cpu().numpy()


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


# --- run discovery -------------------------------------------------------
def list_runs():
    base = HERE / "results"
    if not base.exists():
        return []
    return sorted([d.name for d in base.iterdir() if d.is_dir()], reverse=True)


def load_payload(run_name):
    if not run_name:
        return None
    p = HERE / "results" / run_name / "payload.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


# --- Demo (A): single example -------------------------------------------
def example_plot(length, seed_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = int(length)
    s = int(seed_idx)
    li = TASK.LENGTH_SWEEP.index(L)
    idx = li * TASK.N_SEEDS + s

    pattern = BATCH.patterns[idx]
    embed = BATCH.embeds[idx]
    residual = BATCH.residuals[idx]
    labels = BATCH.labels[idx]

    logits = model_fn(pattern, embed, residual)
    attn = _softmax(logits.astype(np.float64))
    N = len(attn)
    x = np.arange(N)
    match_pos = np.where(labels)[0]

    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.fill_between(x, 0, attn, color="#3b6fb6", alpha=0.5, label="attention")
    ax.plot(x, attn, color="#1f4e8c", lw=1)
    if len(match_pos):
        ax.scatter(match_pos, attn[match_pos], color="crimson", s=55, zorder=5,
                   label="true match-end")
    ax.axhline(1.0 / N, color="gray", ls="--", lw=1, label="uniform 1/N (decision threshold)")
    patt = ["*" if int(p) == -1 else int(p) for p in pattern]
    ax.set_title(f"L={L}  pattern={patt}   (red dots = where the pattern finishes matching)")
    ax.set_xlabel("sequence position")
    ax.set_ylabel("attention weight")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


# --- Demo (B): sweep summary --------------------------------------------
def sweep_plot(run_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    payload = load_payload(run_name)
    if payload is None:
        # Fall back to a live computation so the panel is never empty.
        payload = TASK.evaluate(model_fn)

    Ls = [r["length"] for r in payload["sweep"]]
    sharp = [r["match_sharpness"] for r in payload["sweep"]]
    fpr = [r["false_positive_rate"] for r in payload["sweep"]]
    fnr = [r["false_negative_rate"] for r in payload["sweep"]]
    base = [r["match_sharpness"] for r in payload["linear_baseline"]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))

    a1.plot(Ls, sharp, "o-", color="#1f4e8c", lw=2, label="multi-head AND (this)")
    a1.plot(Ls, base, "s--", color="#b03030", lw=2, label="linear baseline (last token)")
    a1.set_ylim(-0.02, 1.05)
    a1.set_xlabel("pattern length L (composition depth)")
    a1.set_ylabel("match_sharpness")
    a1.set_title("Sharpness vs composition depth")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.25)
    robust = sharp[-1] / sharp[0] if sharp[0] > 1e-12 else 0.0
    a1.annotate(f"length_robustness = {min(1.0, robust):.3f}",
                xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                fontsize=9, color="#1f4e8c")

    a2.plot(Ls, fpr, "o-", color="#d2691e", lw=2, label="false positive rate")
    a2.plot(Ls, fnr, "s-", color="#2e8b57", lw=2, label="false negative rate")
    a2.set_ylim(-0.02, max(0.3, max(fpr + fnr) * 1.2))
    a2.set_xlabel("pattern length L")
    a2.set_ylabel("rate")
    a2.set_title("Error rates vs composition depth")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.25)
    fig.tight_layout()
    return fig


with gr.Blocks() as demo:
    gr.Markdown("# Attention Regex — pass_2: multi-head AND matcher")
    gr.Markdown(
        "One attention head per concrete pattern offset gathers the required "
        "neighbour via a sharp relative-position bias; heads combine with "
        "**min (logical AND)**. A window is a match-end only if *every* offset matches."
    )

    with gr.Tab("Demo"):
        gr.Markdown("### A. Single example — attention lands on true match-ends")
        with gr.Row():
            length_dd = gr.Dropdown(
                choices=[str(L) for L in TASK.LENGTH_SWEEP],
                value=str(TASK.CANONICAL_LENGTH), label="pattern length L")
            seed_sl = gr.Slider(0, TASK.N_SEEDS - 1, step=1, value=0, label="seed index")
        ex_plot = gr.Plot(label="per-position attention")

        gr.Markdown(
            "### B. Sweep summary — robustness across composition depth\n"
            "Left: sharpness vs L for the matcher and the no-composition baseline "
            "(the gap is the lift; the ratio sharp(L=6)/sharp(L=1) is the headline "
            "`length_robustness`). Right: false-positive / false-negative rates — "
            "min/AND keeps FPR flat where summing would leak as L grows."
        )
        run_dd = gr.Dropdown(choices=list_runs(),
                             value=(list_runs()[0] if list_runs() else None),
                             label="results run (latest first)")
        sweep_pl = gr.Plot(label="sweep summary")

        length_dd.change(example_plot, [length_dd, seed_sl], ex_plot)
        seed_sl.change(example_plot, [length_dd, seed_sl], ex_plot)
        run_dd.change(sweep_plot, run_dd, sweep_pl)
        demo.load(example_plot, [length_dd, seed_sl], ex_plot)
        demo.load(sweep_plot, run_dd, sweep_pl)

    with gr.Tab("Benchmark"):
        gr.Markdown("## Benchmark history across attempts")
        benchmark_panel(str(GOAL_DIR))


if __name__ == "__main__":
    demo.launch()
