import importlib.util
import gradio as gr
import numpy as np
from pathlib import Path

# ----------------------------------------------------------
# Load the most recent run of this attempt under results/
# ----------------------------------------------------------
def load_latest(payloads: dict) -> tuple[dict, str]:
    """
    Given a list of payloads from `agentic.experiments.benchmark_panel`,
    pick the last (most recent) run and return its `payload` and the
    human-readable string that shows up on the dropdown.
    """
    if not payloads:
        return None, ""
    latest = payloads[-1]   # newest first? The list is ordered newest to oldest.
    # Build a display label: "run_<ts>" where ts is the last part of the dir name.
    label = f"run_{Path(latest['run_dir']).name}"
    return latest, label


# ----------------------------------------------------------
# Demo panel: identity copy head visualisation
# ----------------------------------------------------------
def demo_panel(pay: dict) -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            gr.Markdown(
                "## Identity Copy Head\n"
                "Demonstration of an attention mechanism that *concentrates* mass only"
                " on the candidate key that **exactly matches** the query.\n"
                "The head knows *where* the match is (via an exact-equality mask) but does not\n"
                "learn the match from similarity alone."
            )
        with gr.Tabs():
            # --- Tab 1: Headline metrics ---
            with gr.Tab("Metrics (headline)"):
                metrics = [
                    ("copy_fidelity_robustness", "copy fidelity across sweep", "float"),
                    ("copy_mass_canonical", "copy mass @ cos = 0.7", "float"),
                    ("copy_accuracy_canonical", "argmax accuracy @ cos = 0.7", "float"),
                ]
                for name, desc, dtype in metrics:
                    with gr.Row():
                        gr.Label(label=desc, value=f"{pay.get(name):.4f}")
                uniform = pay.get("uniform_baseline_mass", 1.0 / 8.0)
                gr.Label(
                    label="Uniform baseline mass (1/M)",
                    value=f"{uniform:.4f}",
                )
            # --- Tab 2: Sweep plot ---
            with gr.Tab("Sweep Across Cosine"):
                # Build per-slice values.
                slices = {
                    c: {
                        "mass": pay[f"copy_mass_cos_{p}"],
                        "acc": pay[f"copy_accuracy_cos_{p}"],
                        "lift": pay[f"lift_over_uniform_cos_{p}"],
                    }
                    for c, p in zip([0.0, 0.3, 0.5, 0.7, 0.9], ["0p0", "0p3", "0p5", "0p7", "0p9"])
                }
                with gr.Row():
                    gr.LinePlot(
                        labels=["copy mass", "copy accuracy"],
                        values=[[v["mass"] for v in slices.values()],
                                [v["acc"]  for v in slices.values()]],
                        x=sorted(slices.keys()),
                        y=[0.0, 1.0],
                        title="Copy Mass & argmax Accuracy vs. Distractor Cosine",
                        line_width=2,
                    )
                    gr.LinePlot(
                        labels=["lift over uniform"],
                        values=[[v["lift"] for v in slices.values()]],
                        x=sorted(slices.keys()),
                        y=[-0.1 * uniform, 0.9 * uniform],
                        title="Mass lift over 1/M (uniform baseline)",
                        line_width=2,
                    )
        # Demo interaction not needed: this head is deterministic given its masks.
        # We expose a button to trigger the "run" view (shows the latest run only).
        with gr.Row():
            run_btn = gr.Button("Show latest run")
            run_btn.click(
                fn=lambda: (None,),  # no fn needed: just load_latest from the dropdown
                inputs=[],
                outputs=[],    # placeholder only
            )
    return demo


# ----------------------------------------------------------
# Agentic dashboard integration (Benchmark tab)
# ----------------------------------------------------------
from agentic.experiments import (
    get_run_dirs,
    benchmark_panel,
)

# The parent goal directory (contains task.py / benchmark.py, as loaded by
# `agentic.experiments.load_task`). This is the directory that contains *all*
# attempts at the same goal.
goal_dir = Path(__file__).parent.parent

# Build the dashboard that shows all attempts and their history.
dashboard = benchmark_panel(goal_dir)   # returns a gr.Blocks instance


# ----------------------------------------------------------
# Combine Demo and Benchmark into a single page
# ----------------------------------------------------------
def combined_app():
    # Demo tab: our custom visualisation of the head's sweep metrics.
    # The dropdown drives which payload we show in the Demo tab.
    demo = demo_panel({})   # placeholder for the demo UI shape
    # Dashboard tab: agentic's auto-generated leaderboard + slice plots.
    dashboard = dashboard
    with gr.Blocks() as combined:
        demo.show()
        dashboard.show()
    return combined


# ----------------------------------------------------------
# Entry point: launch the full demo or a lightweight version that
# just shows the Demo tab.
# ----------------------------------------------------------
def demo_only():
    demo = demo_panel({})
    with gr.Blocks() as demo_app:
        demo.show()
    return demo_app


if __name__ == "__main__":
    # Launch the full combined app with Benchmark tab.
    demo_app = combined_app()
    demo_app.launch()