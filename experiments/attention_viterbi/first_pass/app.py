import gradio as gr
import numpy as np
import json
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir

GOAL_DIR = Path(__file__).parent.parent
RESULTS_DIR = results_dir(__file__).parent  # .../first_pass/results/


def list_runs():
    runs = sorted([d for d in RESULTS_DIR.iterdir() if d.is_dir()], reverse=True)
    return [d.name for d in runs]


def load_payload(run_name):
    run_path = RESULTS_DIR / run_name
    bench_path = run_path / "benchmark.json"
    payload_path = run_path / "payload.json"  # we'll save the full payload in main.py
    if bench_path.exists():
        with open(bench_path) as f:
            bench = json.load(f)
    else:
        bench = {}
    if payload_path.exists():
        with open(payload_path) as f:
            payload = json.load(f)
    else:
        payload = {}
    return bench, payload


def plot_per_head(payload):
    """Bar chart of excess per head (8 bars)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_head = payload.get("per_head", [])
    if not per_head:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No per_head data", ha="center", va="center")
        return fig

    labels = [f"L{ph['layer']}H{ph['head']}" for ph in per_head]
    values = [ph["excess"] for ph in per_head]
    best = payload.get("best_head", {})
    best_label = f"L{best.get('layer',-1)}H{best.get('head',-1)}"

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = ["C1" if f"L{ph['layer']}H{ph['head']}" == best_label else "C0" for ph in per_head]
    bars = ax.bar(labels, values, color=colors, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("Excess attention on predecessor (t-1)")
    ax.set_title("Per-head excess attention (canonical eval set)")
    ax.set_ylim(min(0, min(values)) - 0.05, max(values) + 0.05 if values else 0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    return fig


def plot_positional(payload):
    """Line plot of excess vs query position for best head."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positional = payload.get("positional", [])
    if not positional:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No positional data", ha="center", va="center")
        return fig

    positions = [p["pos"] for p in positional]
    excess = [p["excess"] for p in positional]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(positions, excess, "o-", color="C1", label="Best head")
    ax.axhline(0, color="k", linewidth=0.8, label="Uniform baseline")
    ax.set_xlabel("Query position t")
    ax.set_ylabel("Excess attention on t-1")
    ax.set_title("Excess attention by query position (best head)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_attention_heatmap(payload, run_name, layer, head, seq_idx=0):
    """Heatmap of attention weights for one head on one sequence."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # We need the raw attention weights; they're not in payload.json by default.
    # Try to load from a saved .npy if we stored it, else show placeholder.
    run_path = RESULTS_DIR / run_name
    attn_path = run_path / f"attn_weights.npy"
    if attn_path.exists():
        attn = np.load(attn_path)  # [batch, n_layers, n_heads, seq_len, seq_len]
        if attn.ndim == 5:
            mat = attn[seq_idx, layer, head]
        else:
            mat = None
    else:
        mat = None

    fig, ax = plt.subplots(figsize=(6, 5))
    if mat is not None:
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_title(f"Run {run_name} | Layer {layer} Head {head} | Seq {seq_idx}")
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        plt.colorbar(im, ax=ax, label="Attention weight")
    else:
        ax.text(0.5, 0.5, "Attention weights not saved\n(re-run with --save-attn)", ha="center", va="center")
        ax.set_title("Attention heatmap")
    plt.tight_layout()
    return fig


def demo_tab(run_dropdown):
    bench, payload = load_payload(run_dropdown)
    if not payload:
        return None, None, None, "{}"

    per_head_fig = plot_per_head(payload)
    pos_fig = plot_positional(payload)

    # Default heatmap: best head, first sequence
    best = payload.get("best_head", {"layer": 0, "head": 0})
    heatmap_fig = plot_attention_heatmap(payload, run_dropdown, best["layer"], best["head"], 0)

    metrics_str = json.dumps(bench, indent=2)
    return per_head_fig, pos_fig, heatmap_fig, metrics_str


def on_run_change(run_name):
    per_head_fig, pos_fig, heatmap_fig, metrics_str = demo_tab(run_name)
    return per_head_fig, pos_fig, heatmap_fig, metrics_str


def on_heatmap_controls(run_name, layer, head, seq_idx):
    _, payload = load_payload(run_name)
    heatmap_fig = plot_attention_heatmap(payload, run_name, layer, head, seq_idx)
    return heatmap_fig


# ------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------
with gr.Blocks(title="attention_viterbi — first_pass") as demo:
    gr.Markdown("# attention_viterbi — first_pass\nTrained 2L-4H attention-only transformer on HMM sequences; measuring Viterbi predecessor attention.")

    with gr.Tab("Demo"):
        runs = list_runs()
        default_run = runs[0] if runs else None

        run_dd = gr.Dropdown(choices=runs, value=default_run, label="Run", interactive=True)
        with gr.Row():
            per_head_plot = gr.Plot(label="Per-head excess")
            pos_plot = gr.Plot(label="Positional excess (best head)")
        with gr.Row():
            layer_sl = gr.Slider(0, 1, step=1, value=0, label="Layer")
            head_sl = gr.Slider(0, 3, step=1, value=0, label="Head")
            seq_sl = gr.Slider(0, 99, step=1, value=0, label="Sequence index")
        heatmap_plot = gr.Plot(label="Attention heatmap (query × key)")
        metrics_box = gr.Code(label="Benchmark metrics (benchmark.json)", language="json")

        # Event handlers INSIDE the Blocks context
        run_dd.change(
            fn=on_run_change,
            inputs=[run_dd],
            outputs=[per_head_plot, pos_plot, heatmap_plot, metrics_box],
        )
        for ctrl in [layer_sl, head_sl, seq_sl]:
            ctrl.change(
                fn=on_heatmap_controls,
                inputs=[run_dd, layer_sl, head_sl, seq_sl],
                outputs=[heatmap_plot],
            )
        demo.load(
            fn=on_run_change,
            inputs=[run_dd],
            outputs=[per_head_plot, pos_plot, heatmap_plot, metrics_box],
        )

    with gr.Tab("Benchmark"):
        # Drops in the shared benchmark panel (leaderboard + history across attempts)
        benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()