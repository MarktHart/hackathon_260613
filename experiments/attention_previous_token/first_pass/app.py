"""Gradio app for attention_previous_token first_pass attempt."""
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


def get_run_choices() -> list[str]:
    """Get display names for run dropdown."""
    runs = list_runs()
    if not runs:
        return ["No runs found"]
    return [f"{r.name} (max_attn={load_payload(r)['prev_token_attention']['max_head_value']:.3f})" for r in runs]


def plot_head_heatmap(payload: dict) -> np.ndarray:
    """Create a 2D array [n_layers, n_heads] of prev-token attention values."""
    n_layers = payload["model_info"]["n_layers"]
    n_heads = payload["model_info"]["n_heads"]
    heatmap = np.zeros((n_layers, n_heads), dtype=np.float32)

    for rec in payload["prev_token_attention"]["per_head"]:
        layer = rec["layer"]
        head = rec["head"]
        val = rec["mean_prev_token_attn"]
        heatmap[layer, head] = val

    return heatmap


def format_head_table(payload: dict) -> list[list]:
    """Format per-head data for a Dataframe."""
    rows = []
    for rec in payload["prev_token_attention"]["per_head"]:
        rows.append([
            rec["layer"],
            rec["head"],
            f"{rec['mean_prev_token_attn']:.4f}",
        ])
    return rows


def update_demo(run_choice: str):
    """Update demo tab when run selection changes."""
    runs = list_runs()
    if not runs or run_choice == "No runs found":
        return (
            gr.update(value=None),
            gr.update(value=[["No data", "", ""]]),
            gr.update(value="No run selected"),
        )

    idx = get_run_choices().index(run_choice)
    run_dir = runs[idx]
    payload = load_payload(run_dir)

    if payload is None:
        return (
            gr.update(value=None),
            gr.update(value=[["Error loading", "", ""]]),
            gr.update(value="Failed to load payload"),
        )

    heatmap = plot_head_heatmap(payload)
    table = format_head_table(payload)
    summary = (
        f"**Model:** {payload['model_info']['model_name']}  \n"
        f"**Layers:** {payload['model_info']['n_layers']}, **Heads:** {payload['model_info']['n_heads']}  \n"
        f"**Max prev-token attention:** {payload['prev_token_attention']['max_head_value']:.4f}  \n"
        f"**Mean prev-token attention:** {payload['prev_token_attention']['mean_head_value']:.4f}  \n"
        f"**Uniform baseline:** {payload['uniform_baseline']:.4f}  \n"
        f"**Lift over uniform:** {payload['prev_token_attention']['max_head_value'] / payload['uniform_baseline']:.2f}x"
    )

    return (
        gr.update(value=heatmap),
        gr.update(value=table),
        gr.update(value=summary),
    )


with gr.Blocks(title="Attention Previous Token - first_pass") as demo:
    gr.Markdown("# Attention Previous Token: First Pass (GPT-2 Small)")
    gr.Markdown(
        "Measuring how strongly each attention head attends to the previous token "
        "(position i-1 when processing position i) on random synthetic sequences."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dropdown = gr.Dropdown(
                    choices=get_run_choices(),
                    label="Select run",
                    value=get_run_choices()[0] if get_run_choices() else "No runs found",
                )

            with gr.Row():
                with gr.Column(scale=2):
                    heatmap_plot = gr.Image(label="Previous-token attention per head (layer × head)", type="numpy")
                with gr.Column(scale=1):
                    summary_md = gr.Markdown("Select a run to see details")

            gr.Markdown("### Per-head values")
            head_table = gr.Dataframe(
                headers=["Layer", "Head", "Mean prev-token attn"],
                datatype=["number", "number", "str"],
                row_count=(12 * 12, "fixed"),
                col_count=(3, "fixed"),
            )

            run_dropdown.change(
                fn=update_demo,
                inputs=[run_dropdown],
                outputs=[heatmap_plot, head_table, summary_md],
            )

            # Initial load
            demo.load(
                fn=update_demo,
                inputs=[run_dropdown],
                outputs=[heatmap_plot, head_table, summary_md],
            )

        with gr.Tab("Benchmark"):
            # This panel scans all attempts under the goal and shows leaderboard + history
            benchmark_panel(str(GOAL_DIR)).render()


if __name__ == "__main__":
    demo.launch()