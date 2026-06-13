"""Gradio app for the hand-built synthetic previous-token head model."""
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
    """Return a list of dictionaries for every head's values."""
    rows = []
    for h, v in enumerate(payload["prev_token_attention"]["per_head_mean"]):
        rows.append({
            "head": h,
            "value": f"{v:.4f}"
        })
    return rows


def update_demo(run_choice: str):
    """Update demo tab when run selection changes."""
    runs = list_runs()
    if not runs or run_choice == "No runs found":
        return (
            gr.update(value=None),
            gr.update(value="Select a run to see its payload")
        )

    idx = [r.name for r in runs].index(run_choice)
    run_dir = runs[idx]
    payload = load_payload(run_dir)

    if payload is None:
        return (
            gr.update(value="Failed to load payload from benchmark.json"),
            gr.update(value=None)
        )

    best_head = payload["prev_token_attention"]["per_head_mean"][0]
    uniform_baseline = payload["uniform_baseline"]
    lift = best_head - uniform_baseline
    ratio = f"{best_head / uniform_baseline:.2f}x"

    markdown = (
        f"**Model:** *hand-built previous-token circuit*  \n"
        f"**Best previous-token head attention:** {best_head:.4f}  \n"
        f"**Uniform (no-signal) baseline:** {uniform_baseline:.6f}  \n"
        f"**Lift over baseline:** {lift:.4f} (additive)  \n"
        f"**Ratio:** {ratio} (倍以上 chance)" if uniform_baseline > 0 else ratio
    )

    head_table = gr.DataFrame(
        headers=["Head", "Previous-token attention"],
        datatype=["number", "str"],
        value=get_head_summary(payload),
        column_widths=["20%", "60%"]
    )
    return (
        gr.update(value=markdown),
        gr.update(value=head_table)
    )


with gr.Blocks(title="Synthetic previous-token head demo") as demo:
    gr.Markdown("# Hand-built synthetic circuit: one previous-token head")
    gr.Markdown(
        "This model contains exactly one attention head that strongly attends to the immediately preceding token, while all other heads emit row-wise uniform distributions. No transformer, no pre-trained weights — just a minimal synthetic circuit."
    )

    with gr.Tabs():
        with gr.Tab("Demo"):
            with gr.Row():
                run_dropdown = gr.Dropdown(
                    choices=[r.name for r in list_runs()] + ["No runs found"],
                    label="Select run",
                    value=list_runs()[-1].name if list_runs() else "No runs found"
                )
                desc = gr.Markdown("Summary and per-head breakdown of the selected run:")

            with gr.Row():
                summary_md = gr.Markdown("Select a run to see its payload")
                per_head_table = gr.Dataframe(
                    headers=["Head", "Previous-token attention"],
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