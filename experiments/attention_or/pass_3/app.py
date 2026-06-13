import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir

GOAL_DIR = Path(__file__).parent.parent


def load_payload(run_dir: Path):
    payload_path = run_dir / "benchmark.json"
    if payload_path.exists():
        with open(payload_path) as f:
            return json.load(f)
    return None


def get_run_choices():
    results_root = Path(__file__).parent / "results"
    if not results_root.exists():
        return []
    return [str(d) for d in sorted(results_root.iterdir())]


def make_figs(payload: dict):
    """Create matplotlib figures for the Demo tab."""
    if not payload or "sweep" not in payload:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No data yet — run main.py first", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig, fig

    sweep = payload["sweep"]
    cos_vals = np.array([s["cos"] for s in sweep])

    # Figure 1: OR sharpness vs cos(q_A, q_B) with linear baseline
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    or_sharp = [s.get("or_sharpness", 0) for s in sweep]  # will compute below
    # Recompute from raw scores for accuracy
    or_sharp = []
    lin_sharp = []
    for s in sweep:
        denom = max(s["s_A_at_A"], s["s_B_at_B"])
        num_or = min(s["s_AB_at_A"], s["s_AB_at_B"])
        or_sharp.append(num_or / denom if denom > 0 else 0.0)
        s_lin_A = s["s_A_at_A"] + s["s_B_at_A"]
        s_lin_B = s["s_A_at_B"] + s["s_B_at_B"]
        num_lin = min(s_lin_A, s_lin_B)
        lin_sharp.append(num_lin / denom if denom > 0 else 0.0)

    ax1.plot(cos_vals, or_sharp, 'o-', label="OR head (hand-built max-pooling)", linewidth=2, color="#4e79a7")
    ax1.plot(cos_vals, lin_sharp, 's--', label="Linear superposition (s_A + s_B)", linewidth=2, color="#f28e2b")
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5, label="Ideal OR (sharpness=1)")
    ax1.set_xlabel(r"Cos(q_A, q_B)")
    ax1.set_ylabel("Sharpness  =  min(s_AB@A, s_AB@B) / max(s_A@A, s_B@B)")
    ax1.set_title("OR Sharpness across feature overlap")
    ax1.set_ylim(-0.05, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Figure 2: Noise leakage for q_AB
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    noise_leak = []
    for s in sweep:
        denom = max(s["s_AB_at_A"], s["s_AB_at_B"])
        noise_leak.append(s["s_AB_noise_max"] / denom if denom > 0 else 0.0)

    ax2.plot(cos_vals, noise_leak, '^-', label="Noise leakage from q_AB", linewidth=2, color="#b07aa1")
    ax2.set_xlabel(r"Cos(q_A, q_B)")
    ax2.set_ylabel("Noise leakage  =  max_noise / max_signal")
    ax2.set_title(r"Mass leaked to the n-2 noise keys for q_AB")
    ax2.set_ylim(-0.02, max(0.5, max(noise_leak)*1.2) if noise_leak else 0.5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    return fig1, fig2


def update_demo(run_dir_str: str):
    run_dir = Path(run_dir_str)
    payload = load_payload(run_dir)
    return make_figs(payload)


with gr.Blocks() as demo:
    gr.Markdown("# attention_or — pass_3 (hand-built max-pooling OR circuit)")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            run_choices = get_run_choices()
            run_dropdown = gr.Dropdown(
                choices=run_choices,
                value=run_choices[-1] if run_choices else None,
                label="Run directory",
                interactive=True,
            )
            with gr.Row():
                plot1 = gr.Plot(label="OR sharpness vs cos(q_A, q_B)")
                plot2 = gr.Plot(label="Noise leakage from q_AB")

            run_dropdown.change(fn=update_demo, inputs=run_dropdown, outputs=[plot1, plot2])
            demo.load(fn=update_demo, inputs=run_dropdown, outputs=[plot1, plot2])

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt benchmark history")
            benchmark_panel(str(GOAL_DIR))

if __name__ == "__main__":
    demo.launch()