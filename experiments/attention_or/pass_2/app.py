import json
import numpy as np
import gradio as gr
from pathlib import Path

from agentic.experiments import benchmark_panel, results_dir

GOAL_DIR = Path(__file__).parent.parent


def load_latest_payload() -> tuple[dict, Path]:
    """Load payload from the most recent run directory under this attempt."""
    results_root = Path(__file__).parent / "results"
    if not results_root.exists():
        return {}, None
    run_dirs = sorted(results_root.iterdir())
    if not run_dirs:
        return {}, None
    latest = run_dirs[-1]
    payload_path = latest / "benchmark.json"
    if payload_path.exists():
        with open(payload_path) as f:
            return json.load(f), latest
    return {}, latest


def make_demo_plots(payload: dict):
    """Create matplotlib figures for the Demo tab."""
    if not payload or "sweep" not in payload:
        return None, None

    sweep = payload["sweep"]
    cos_vals = np.array([s["cos"] for s in sweep])
    tags = [f"cos={c:.1f}" for c in cos_vals]
    
    # Figure 1: OR sharpness vs cos (also show linear baseline)
    fig1, ax = plt.subplots(figsize=(8, 4.5))
    or_sharpness = [s["or_sharpness_cos_0p0"] for s in sweep]  # recompute from values
    lin_sharpness = [s["linear_baseline_sharpness_cos_0p0"] for s in sweep]
    ax.plot(cos_vals, or_sharpness, 'o-', label="OR head (hand-built max circuit)", linewidth=2, color="#4e79a7")
    ax.plot(cos_vals, lin_sharpness, 's--', label="Linear superposition reference", linewidth=2, color="#f28e2b")
    ax.set_xlabel(r"Cos(q_A, q_B)")
    ax.set_ylabel("Sharpness (min signal / max single-query signal)")
    ax.set_title("OR Sharpness across feature overlap")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Figure 2: Noise leakage for q_AB
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    noise_leakage = [s["or_noise_leakage_cos_0p0"] for s in sweep]
    ax2.plot(cos_vals, noise_leakage, '^-', label="Noise leakage from q_AB (probes mass left of signal)", linewidth=2, color="#b07aa1")
    ax2.set_xlabel(r"Cos(q_A, q_B)")
    ax2.set_ylabel("Noise leakage fraction")
    ax2.set_title(r"Mass leaked to the n-2 noise keys for q_AB")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    return fig1, fig2


def update_demo(run_dir_str: str):
    run_dir = Path(run_dir_str)
    payload_path = run_dir / "benchmark.json"
    if payload_path.exists():
        with open(payload_path) as f:
            payload = json.load(f)
    else:
        payload = {}
    fig1, fig2 = make_demo_plots(payload)
    return fig1, fig2


def get_run_choices():
    results_root = Path(__file__).parent / "results"
    if not results_root.exists():
        return []
    return [str(d) for d in sorted(results_root.iterdir())]


with gr.Blocks() as demo:
    gr.Markdown("# attention_or — pass_2 (hand-set max attention mechanism)")

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

            # Initial load of latest results
            demo.load(fn=update_demo, inputs=run_dropdown, outputs=[plot1, plot2])

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt benchmark history")
            benchmark_panel(str(GOAL_DIR)).render()


if __name__ == "__main__":
    demo.launch()