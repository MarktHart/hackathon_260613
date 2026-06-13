"""
Gradio app for attention_sign_threshold / pass_4 (TRAINED bilinear QK head).

Demo tab:
  1. Faithfulness/ablation sweep -- mean attention vs cos for the trained
     circuit, plus two causal ablations (zero M, off-diagonal-only M) and the
     linear baseline. Zeroing the learned circuit flattens the sweep to 0.5.
  2. Learned-M heatmap -- shows training rediscovered the dot product (M ~ s*I).
  3. Live explorer -- attention distribution at any cosine, recomputed on the
     GPU from the saved learned M.
Benchmark tab: agentic.experiments.benchmark_panel(goal_dir).
"""
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from agentic.experiments import benchmark_panel, results_dir

DEVICE = "cuda"
D = 64
GOAL_DIR = Path(__file__).resolve().parent.parent
RESULTS_ROOT = Path(results_dir(__file__)).parent


def list_runs():
    if not RESULTS_ROOT.exists():
        return []
    return [str(p) for p in sorted((p for p in RESULTS_ROOT.iterdir() if p.is_dir()), reverse=True)]


def _load(run_dir, name):
    p = Path(run_dir) / name
    if not p.exists():
        return None
    if name.endswith(".json"):
        with open(p) as f:
            return json.load(f)
    return np.load(p)


def plot_sweep(run_dir_str):
    if not run_dir_str:
        return None, "No run selected."
    art = _load(run_dir_str, "artifacts.json")
    if art is None:
        return None, "No artifacts.json in run."
    cos = art["cos"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cos, art["mean_trained"], "o-", color="#1f77b4", lw=2.6, label="trained QK head q^T M k")
    ax.plot(cos, art["mean_ablate_zero"], "--", color="#d62728", lw=2.0, label="ablation: M→0 (circuit removed)")
    ax.plot(cos, art["mean_ablate_offdiag"], ":", color="#9467bd", lw=2.0, label="ablation: off-diagonal M only")
    ax.plot(cos, art["linear_baseline"], "--", color="#8c8da0", lw=2.0, label="linear baseline max(0,cos)")
    ax.axvline(0.0, color="black", ls=":", lw=1.2, alpha=0.6)
    ax.axhline(0.5, color="black", ls=":", lw=1.2, alpha=0.6)
    ax.set_xlabel("cos(q, k)", fontsize=12)
    ax.set_ylabel("mean attention weight", fontsize=12)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", fontsize=9)
    ax.set_title("Trained sign detector vs causal ablations", fontsize=13)
    fig.tight_layout()
    align = art.get("M_identity_alignment", float("nan"))
    msg = (f"Faithfulness: learned M has cos(vec(M), vec(I)) = **{align:.3f}** — "
           "training rediscovered the dot product. Zeroing M flattens the sweep to ~0.5 "
           "(no sign detection); the off-diagonal part alone carries no signal. "
           "The sharp flip lives in the learned diagonal circuit, not the metric.")
    return fig, msg


def plot_heatmap(run_dir_str):
    if not run_dir_str:
        return None
    M = _load(run_dir_str, "M.npy")
    if M is None:
        return None
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    v = float(np.abs(M).max())
    im = ax.imshow(M, cmap="RdBu_r", vmin=-v, vmax=v)
    ax.set_title("Learned bilinear form M (≈ s·Identity)", fontsize=12)
    ax.set_xlabel("key dim")
    ax.set_ylabel("query dim")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig


def gen_pairs_at_cos(cosine, n, seed):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(n, D)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-8
    o = rng.normal(size=(n, D)).astype(np.float32)
    o -= np.sum(o * q, axis=1, keepdims=True) * q
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-8
    sin = np.sqrt(max(0.0, 1.0 - cosine * cosine))
    k = cosine * q + sin * o
    k /= np.linalg.norm(k, axis=1, keepdims=True) + 1e-8
    return q, k


def plot_explorer(run_dir_str, cosine, n_pairs, seed):
    if not run_dir_str:
        return None, "No run selected."
    M = _load(run_dir_str, "M.npy")
    if M is None:
        return None, "No M.npy in run."
    Mt = torch.as_tensor(M, dtype=torch.float32, device=DEVICE)
    q, k = gen_pairs_at_cos(float(cosine), int(n_pairs), int(seed))
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(k, dtype=torch.float32, device=DEVICE)
    logit = torch.einsum("bd,de,be->b", qt, Mt, kt)
    attn = torch.sigmoid(logit).detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(attn, bins=np.linspace(0, 1, 31), color="#1f77b4", alpha=0.8)
    ax.axvline(0.5, color="black", ls=":", lw=1.4, label="decision = 0.5")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("attention weight", fontsize=11)
    ax.set_ylabel("count", fontsize=11)
    tgt = "attend" if cosine > 0 else "anti-attend" if cosine < 0 else "boundary"
    ax.set_title(f"Trained head attention at cos={float(cosine):+.2f}  (target: {tgt})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    msg = f"mean attention = {attn.mean():.3f}"
    return fig, msg


_runs = list_runs()
_default = _runs[0] if _runs else None

with gr.Blocks(title="Attention Sign Threshold — Trained QK Circuit") as demo:
    gr.Markdown(
        "# Attention Sign Threshold — Trained bilinear QK head + causal ablation\n"
        "A **trained** single attention head, score = `q^T M k` (one learned d×d matrix, "
        "no MLP — the smallest delta from `base_model.py`). Training on a sign objective "
        "**rediscovers the dot product** (M ≈ s·I) and a sharp flip at cos = 0."
    )

    with gr.Tab("Demo"):
        run_dd = gr.Dropdown(choices=_runs, value=_default, label="Run")

        gr.Markdown("### 1 · Trained sweep vs causal ablations")
        sweep_plot = gr.Plot()
        sweep_msg = gr.Markdown()

        gr.Markdown("### 2 · Learned M (should look like a scaled identity)")
        heat_plot = gr.Plot()

        gr.Markdown("### 3 · Live single-cosine explorer (from learned M, on GPU)")
        with gr.Row():
            cos_slider = gr.Slider(-1.0, 1.0, value=0.0, step=0.05, label="cos(q, k)")
            n_slider = gr.Slider(100, 3000, value=800, step=100, label="# pairs")
            seed_box = gr.Number(value=7, precision=0, label="seed")
        explorer_plot = gr.Plot()
        explorer_msg = gr.Markdown()

        def refresh_all(run_dir, cos, n, seed):
            sf, sm = plot_sweep(run_dir)
            ef, em = plot_explorer(run_dir, cos, n, seed)
            return sf, sm, plot_heatmap(run_dir), ef, em

        def refresh_explorer(run_dir, cos, n, seed):
            return plot_explorer(run_dir, cos, n, seed)

        run_dd.change(refresh_all, inputs=[run_dd, cos_slider, n_slider, seed_box],
                      outputs=[sweep_plot, sweep_msg, heat_plot, explorer_plot, explorer_msg])
        for ctl in (cos_slider, n_slider, seed_box):
            ctl.change(refresh_explorer, inputs=[run_dd, cos_slider, n_slider, seed_box],
                       outputs=[explorer_plot, explorer_msg])
        demo.load(refresh_all, inputs=[run_dd, cos_slider, n_slider, seed_box],
                  outputs=[sweep_plot, sweep_msg, heat_plot, explorer_plot, explorer_msg])

    with gr.Tab("Benchmark"):
        benchmark_panel(GOAL_DIR)


if __name__ == "__main__":
    demo.launch()
