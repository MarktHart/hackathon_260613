import gradio as gr
import json
import os
from pathlib import Path
from typing import Any, Dict

# ------------------------------------------------------------
# Imports from goal directory (syndir layout)
# ------------------------------------------------------------
goal_parent = Path(__file__).parent.parent
task_source = str(goal_parent / "experiments" / "attention_block_2d" / "task.py")
task_module = gr.load_module(task_source)
task: Any = task_module

# ------------------------------------------------------------
# Result loading helpers
# ------------------------------------------------------------
def _get_results_dir(*args) -> Path:
    module_path = args[0]
    module_dir = Path(module_path).parent
    results_dir = module_dir / "results"
    assert results_dir.is_dir(), f"No results directory under {results_dir}"
    # get newest "run-*" directory under results
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0]  # newest run


def _load_latest_benchmark(attempt: str) -> Dict:
    base_dir = goal_parent / "experiments" / "attention_block_2d"
    results_dir = Path(base_dir) / attempt / "results"
    bmark_path = results_dir / "benchmark.json"
    assert bmark_path.is_file(), f"benchmark.json missing at {bmark_path}"
    return json.loads(bmark_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# Demo tab: visualise a single attention matrix (first of batch)
# ------------------------------------------------------------
def _demo_content() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            attempt_dd = gr.Dropdown(
                label="Attempt",
                value="pass_3",
                choices=[d.name for d in (goal_parent / "experiments" / "attention_block_2d").iterdir() if d.is_dir()],
                interactive=True,
            )
            # metrics block (bar chart of per-pattern accuracy)
            metric_plot = gr.Plot(label="Per-family accuracy")

        with gr.Row():
            # heatmap of a single attention matrix from the run directory
            heatmap = gr.Image(label="Row-stochastic attention matrix (8×8 grid)")

        def _render_demo(selected_attempt: str):
            payload = _load_latest_benchmark(selected_attempt)

            # Bar chart of per-pattern accuracy across all records
            import plotly.express as px
            patterns = ["local", "dilated", "global", "causal_2d"]
            accuracies = [
                payload["sweep"][i].get("correct", False) for i in range(16)
            ]
            scores = {
                "local": sum(accuracies[i] for i in range(0, 4)),
                "dilated": sum(accuracies[i] for i in range(4, 8)),
                "global": sum(accuracies[i] for i in range(8, 12)),
                "causal_2d": sum(accuracies[i] for i in range(12, 16)),
            }
            x_labels = [label + (f" ({count}/4)" if count >= 4 else f" ({count}/4)") for label, count in scores.items()]
            y_vals = [scores[k] / 4 for k in patterns]
            fig = px.bar(
                x=x_labels,
                y=y_vals,
                labels={"x": "Pattern family", "y": "Accuracy"},
                title="Per-family accuracy on 16 canonical examples",
                text=[f"{v:.1%}" for v in y_vals],
            )
            fig.update_traces(texttemplate='%{text}', textposition='outside')
            metric_plot.plot(fig)

            # Load the saved attention tensor from the run directory (optional artifact)
            attn_saved_path = Path(goal_parent / "experiments" / "attention_block_2d") / selected_attempt / "results" / "run-" / "model_attn.npy"
            if attn_saved_path.is_file():
                attn = np.load(attn_saved_path.name)  # shape [16, N, N]; take first matrix
                heatmap.set(attn[0].reshape(8, 8))   # show first example only
            else:
                # fall back to a synthetic local matrix
                A = np.zeros((64, 64))
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        r_k, c_k = 0 + dr, 0 + dc  # query at (0,0)
                        if 0 <= r_k < 8 and 0 <= c_k < 8:
                            A[0, r_k * 8 + c_k] = 1.0
                A = A / A.sum(1, keepdims=True)[0]  # row-stochastic
                heatmap.set(A.reshape(8, 8))
            return metric_plot, heatmap

        attempt_dd.change(
            fn=_render_demo,
            inputs=attempt_dd,
            outputs=[metric_plot, heatmap],
        )
    return demo


# ------------------------------------------------------------
# Benchmark panel: leaderboard across all attempts
# ------------------------------------------------------------
def _benchmark_panel() -> gr.Blocks:
    panel = gr.Blocks()
    with penel:
        task.benchmark_panel(
            str(goal_parent),
            demo_tab="Demo",
            show_history=True,
        )
    return panel


def create_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Tab("Demo"):
            _demo_content().render()
        with gr.Tab("Benchmark"):
            _benchmark_panel().render()
    return demo


demo = create_demo()

if __name__ == "__main__":
    demo.launch()