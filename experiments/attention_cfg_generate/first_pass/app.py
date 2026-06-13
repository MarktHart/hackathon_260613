import gradio as gr
from agentic.experiments import benchmark_panel, load_task, results_dir

# Load the goal's task file (contains generate, evaluate, payload contract)
task = load_task(__file__)

# The actual attempt function lives in main.py; import it here
from main import model_fn as attempt_model_fn

def demo_metrics():
    """
    Runs the attempt model on the canonical batch and returns the raw payload.
    Intended for the Demo tab's "Run" button.
    """
    batch = task.generate(seed=42)
    payload = task.evaluate(attempt_model_fn)
    return str(payload)

# ---- Gradio app ----
with gr.Blocks() as demo:
    # Demo Tab: Run the attempt and display key metrics
    with gr.Blocks():
        gr.Markdown("### Run `first_pass` hand-built model")
        run_btn = gr.Button("Run", variant="primary")
        result = gr.Markdown()
        run_btn.click(demo_metrics, outputs=[result])

    # Benchmark Tab: Leaderboard history panel from the experiment
    with gr.Blocks():
        gr.Markdown("### Leaderboard and history across attempts")
        # This scans all attempts under this goal and plots the history
        benchmark_panel(goal_dir="experiments/attention_cfg_generate")

# Default to the Demo tab
demo.set_theme("default")
if __name__ == "__main__":
    demo.launch()