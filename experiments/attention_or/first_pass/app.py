import json
from pathlib import Path
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from agentic.experiments import benchmark_panel

# Goal directory (parent of this attempt)
GOAL_DIR = Path(__file__).parent.parent


def load_latest_payload() -> tuple[dict, Path]:
    """Load payload from the most recent run directory."""
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
    rhos = [s["rho"] for s in sweep]
    states = ["00", "01", "10", "11"]
    colors = {"00": "tab:red", "01": "tab:blue", "10": "tab:green", "11": "tab:purple"}

    # Figure 1: Mean logit per state vs rho
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    for state in states:
        means = [s[f"mean_{state}"] for s in sweep]
        stds = [s[f"std_{state}"] for s in sweep]
        ax1.errorbar(rhos, means, yerr=stds, label=state, color=colors[state],
                     marker='o', capsize=3, alpha=0.8)
    ax1.set_xlabel(r"Feature cosine similarity $\rho$")
    ax1.set_ylabel("Mean attention logit")
    ax1.set_title("OR logit by input state across $\\rho$")
    ax1.legend(title="State (ab)")
    ax1.grid(True, alpha=0.3)

    # Figure 2: Sharpness vs rho (with linear baseline)
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    sharpness = []
    baseline = []
    for s in sweep:
        rho = s["rho"]
        mean_00 = s["mean_00"]
        on_means = [s["mean_01"], s["mean_10"], s["mean_11"]]
        on_stds = [s["std_01"], s["std_10"], s["std_11"]]
        worst = int(np.argmin(on_means))
        gap = on_means[worst] - mean_00
        denom = 0.5 * (on_stds[worst] + s["std_00"]) + 1e-8
        sharpness.append(gap / denom)
        baseline.append(rho / payload["config"]["sigma"])

    ax2.plot(rhos, sharpness, 'o-', label="OR head (max-over-detectors)", color="tab:blue", linewidth=2)
    ax2.plot(rhos, baseline, 's--', label="Linear baseline (single feature)", color="tab:orange", linewidth=2)
    ax2.set_xlabel(r"Feature cosine similarity $\rho$")
    ax2.set_ylabel("Sharpness (worst-case SNR)")
    ax2.set_title("OR sharpness vs $\\rho$ — robustness to superposition")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    return fig1, fig2


def update_demo(run_dir_str: str):
    """Callback to update plots when run selector changes."""
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
    gr.Markdown("# attention_or — first_pass (max-over-detectors)")

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
                plot1 = gr.Plot(label="Logit distributions per state")
                plot2 = gr.Plot(label="Sharpness vs $\\rho$")

            run_dropdown.change(fn=update_demo, inputs=run_dropdown, outputs=[plot1, plot2])

            # Initial load
            if run_choices:
                demo.load(fn=update_demo, inputs=run_dropdown, outputs=[plot1, plot2])

        with gr.TabItem("Benchmark"):
            gr.Markdown("## Cross-attempt benchmark history")
            benchmark_panel(str(GOAL_DIR)).render()

if __name__ == "__main__":
    demo.launch()