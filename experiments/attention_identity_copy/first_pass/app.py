"""
Gradio app for the first_pass attempt.

Demo tab:
- Sweep curve: copy_mass vs distractor cosine (from benchmark.json)
- Attention heatmap for a handful of trials at a selected cosine
- Interactive temperature slider: recomputes attention on the fly to show
  how τ controls the copy/sharpness trade-off.

Benchmark tab: shared leaderboard across all attempts at this goal.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel, results_dir

# ---------------------------------------------------------------------------
# Utility: find latest run directory
# ---------------------------------------------------------------------------
def _latest_run_dir() -> Path:
    base = results_dir(__file__).parent  # .../first_pass/results/
    runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No run directories found under results/")
    return runs[0]


# ---------------------------------------------------------------------------
# Load artefacts from a run
# ---------------------------------------------------------------------------
def _load_run(run_dir: Path):
    # benchmark.json from record_benchmark
    with open(run_dir / "benchmark.json") as f:
        bench = json.load(f)

    # samples.npz from main.py
    samples = {}
    if (run_dir / "samples.npz").exists():
        data = np.load(run_dir / "samples.npz")
        for key in data.files:
            samples[key] = data[key]

    # config.json
    config = {}
    if (run_dir / "config.json").exists():
        with open(run_dir / "config.json") as f:
            config = json.load(f)

    return bench, samples, config


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _plot_sweep(bench: dict) -> plt.Figure:
    sweep = bench["sweep"]
    cos_vals = [s["cos"] for s in sweep]
    copy_mass = [s["copy_mass"] for s in sweep]
    copy_acc = [s["copy_accuracy"] for s in sweep]
    baseline = bench["uniform_baseline_mass"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(cos_vals, copy_mass, "o-", label="copy_mass (attention on target)")
    ax.plot(cos_vals, copy_acc, "s--", label="copy_accuracy (argmax == target)")
    ax.axhline(baseline, color="gray", linestyle=":", label=f"uniform baseline (1/M={baseline:.3f})")
    ax.set_xlabel("Distractor cosine with query")
    ax.set_ylabel("Metric")
    ax.set_title("Identity Copy Sweep")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _plot_heatmap(attn: np.ndarray, target_idx: np.ndarray, cos: float, temperature: float) -> plt.Figure:
    """Heatmap of attention weights for a few trials."""
    n_trials, n_candidates = attn.shape
    fig, ax = plt.subplots(figsize=(8, max(3, n_trials * 0.35)))

    im = ax.imshow(attn, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    # Mark target positions
    for i, t in enumerate(target_idx):
        ax.plot(t, i, "r*", markersize=12, markeredgecolor="white", markeredgewidth=0.5)

    ax.set_xticks(range(n_candidates))
    ax.set_yticks(range(n_trials))
    ax.set_xlabel("Candidate key index")
    ax.set_ylabel("Trial")
    ax.set_title(f"Attention weights (cos={cos}, τ={temperature}) — ★ = target")
    plt.colorbar(im, ax=ax, label="Attention weight")
    fig.tight_layout()
    return fig


def _recompute_attention(queries: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
    """Scaled dot-product attention with given temperature."""
    logits = queries @ keys.transpose(0, 2, 1) / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
with gr.Blocks(title="Attention Identity Copy — first_pass") as demo:
    gr.Markdown("## Attention Identity Copy — first_pass\n"
                "Scaled dot-product attention with temperature τ. "
                "Query equals target key (cos=1); distractors at cosine `cos`.")

    # --- Run selector ---
    run_dd = gr.Dropdown(
        label="Run",
        choices=[],  # populated on load
        interactive=True,
    )

    # --- Sweep plot (static from benchmark) ---
    sweep_plot = gr.Plot(label="Sweep: copy mass vs distractor cosine")

    # --- Heatmap controls ---
    with gr.Row():
        cos_selector = gr.Dropdown(
            label="Distractor cosine (cos)",
            choices=[0.0, 0.3, 0.5, 0.7, 0.9],
            value=0.7,
            interactive=True,
        )
        temp_slider = gr.Slider(
            minimum=0.05, maximum=1.0, value=0.1, step=0.05,
            label="Temperature τ (lower = sharper copy)",
            interactive=True,
        )

    heatmap_plot = gr.Plot(label="Attention heatmap (first 16 trials)")

    # --- Benchmark tab ---
    with gr.Tab("Benchmark"):
        # Goal directory is two levels up from this attempt's directory
        goal_dir = Path(__file__).parent.parent
        benchmark_panel(str(goal_dir))

    # --- Event handlers (all INSIDE the Blocks context) ---
    def _list_runs():
        base = results_dir(__file__).parent
        runs = sorted(base.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [f"{p.name}  ({p.stat().st_size:,} bytes)" for p in runs], runs[0].name if runs else None

    def _on_load():
        choices, latest = _list_runs()
        return gr.update(choices=choices, value=choices[0] if choices else None), latest

    def _load_run_data(run_label: str):
        if not run_label:
            return None, None, None
        run_name = run_label.split("  (")[0]
        run_dir = results_dir(__file__).parent / run_name
        bench, samples, config = _load_run(run_dir)
        return bench, samples, config

    def _update_sweep(run_label):
        bench, _, _ = _load_run_data(run_label)
        if bench is None:
            return None
        return _plot_sweep(bench)

    def _update_heatmap(run_label, cos, temperature):
        bench, samples, config = _load_run_data(run_label)
        if bench is None:
            return None

        # Load the original queries/keys for this cosine to recompute with new τ
        # We need the raw data; regenerate from task (deterministic seed=0)
        from agentic.experiments import load_task
        task = load_task(__file__)
        batch = task.generate(seed=0)
        sl = next(s for s in batch.slices if abs(s.cos - cos) < 1e-6)
        queries = sl.queries[:16]
        keys = sl.keys[:16]
        target_idx = sl.target_idx[:16]

        # Recompute attention with interactive temperature
        attn = _recompute_attention(queries, keys, temperature)
        return _plot_heatmap(attn, target_idx, cos, temperature)

    # Wire up events
    demo.load(
        _on_load,
        outputs=[run_dd, run_dd],  # second output preselects latest
    )
    run_dd.change(
        _update_sweep,
        inputs=[run_dd],
        outputs=[sweep_plot],
    )
    run_dd.change(
        _update_heatmap,
        inputs=[run_dd, cos_selector, temp_slider],
        outputs=[heatmap_plot],
    )
    cos_selector.change(
        _update_heatmap,
        inputs=[run_dd, cos_selector, temp_slider],
        outputs=[heatmap_plot],
    )
    temp_slider.change(
        _update_heatmap,
        inputs=[run_dd, cos_selector, temp_slider],
        outputs=[heatmap_plot],
    )

if __name__ == "__main__":
    demo.launch()