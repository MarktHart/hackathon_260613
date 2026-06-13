"""Gradio app for the hand-built previous-token head on GPU.

Demo tab shows a markdown summary and a data table of per-noise metrics.
Benchmark tab renders the full leaderboard and metric over time across
all attempts for `attention_previous_token`.
"""

import json
from pathlib import Path
import gradio as gr
import numpy as np

from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent
ATTEMPT_DIR = Path(__file__).parent
RESULTS_DIR = ATTEMPT_DIR / "results"


def list_runs() -> list[Path]:
    """Return sorted list of run directories (newest first)."""
    if not RESULTS_DIR.exists():
        return []
    runs = [d for d in RESULTS_DIR.iterdir() if d.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def load_payload(run_dir: Path) -> dict | None:
    """Load benchmark.json from a run directory."""
    bench_path = run_dir / "benchmark.json"
    if not bench_path.exists():
        return None
    with open(bench_path, "r") as f:
        return json.load(f)


def get_head_summary(payload: dict) -> list[dict]:
    """Extract headline metrics across the noise sweep."""
    rows = []
    for rec in payload["sweep"]:
        noise = rec["noise"]
        prev = rec["prev_token_attention"]
        rows.append({
            "noise": noise,
            "prev_token_attention": f"{prev:.4f}"
        })
    return rows


def update_demo(run_choice: str):
    """Render per-run metrics when the dropdown changes."""
    runs = list_runs()
    if not runs or run_choice == "No runs found":
        return (
            gr.update(value="Select a run to see its payload"),
            gr.update(value=None)
        )

    idx = [r.name for r in runs].index(run_choice)
    run_dir = runs[idx]
    payload = load_payload(run_dir)

    if payload is None:
        return (
            gr.update(value="Failed to load payload from benchmark.json"),
            gr.update(value=None)
        )

    # Grab headline numbers.
    canonical = next(r for r in payload["sweep"] if r["noise"] == 0.0)
    uniform_baseline = payload["uniform_baseline"]
    lift = canonical["prev_token_attention"] - uniform_baseline
    ratio = f"{canonical['prev_token_attention'] / uniform_baseline:.2f}x"

    markdown = (
        f"**Model:** *Hand-built previous-token head on the GPU*  \n"
        f"**Previous-token attention ( canonical noise 0.0):** {canonical['prev_token_attention']:.4f}  \n"
        f"**Uniform baseline:** {uniform_baseline:.6f}  \n"
        f"**Lift over baseline:** {lift:.4f}  \n"
        f"**Signal / baseline ratio:** {ratio}  \n"
    )
    table = gr.DataFrame(
        headers=["Noise level", "Previous-token attention"],
        datatype=["number", "str"],
        value=get_head_summary(payload),
        column_widths=["20%", "60%"]
    )
    return (
        gr.update(value=markdown),
        gr.update(value=table)
    )


with gr.Blocks(title="Previous-Token Head Demo (GPU)") as demo:
    gr.Markdown("# Hand-built previous-token head on GPU")
    gr.Markdown(
        "A synthetic circuit implementing a single previous-token attention head: each query i attends sharply to key i-1. All real computation runs on CUDA; the head produces a (seq_len, seq_len) logits tensor that the evaluator softmaxes and因果 masks. Metrics measured at canonical (0.0) and swept noise levels 0.25, 0.5, 1.0, 2.0."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dropdown = gr.Dropdown(
                    choices=[r.name for r in list_runs()] + ["No runs found"],
                    label="Select run",
                    value=list_runs()[-1].name if list_runs() else "No runs found"
                )
                desc = gr.Markdown("Headline metrics and per-noise breakdown of the selected run:")

            with gr.Row():
                summary_md = gr.Markdown("Select a run to see its payload")
                per_head_table = gr.Dataframe(
                    headers=["Noise level", "Previous-token attention"],
                    datatype=["number", "str"],
                    value=[], column_widths=["20%", "60%"]
                )

            run_dropdown.change(
                fn=update_demo,
                inputs=[run_dropdown],
                outputs=[summary_md, per_head_table]
            )
            demo.load(
                fn=update_demo,
                inputs=[run_dropdown],
                outputs=[summary_md, per_head_table]
            )

        with gr.Tab("Benchmark"):
            benchmark_panel(str(GOAL_DIR)).render()


if __name__ == "__main__":
    demo.launch()