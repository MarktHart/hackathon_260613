import gradio as gr
import json
import os
from pathlib import Path
from typing import Any, Dict
import numpy as np
from agentic.experiments import task_get_metrics

# ------------------------------------------------------------
# Imports from goal directory
# ------------------------------------------------------------
goal_dir = Path(__file__).parent.parent
task_source = str(goal_dir / "experiments" / "attention_block_2d" / "task.py")
task_module = gr.load_module(task_source)
task: Any = task_module
BLOCK_SIZES = task.BLOCK_SIZES  # (1, 2, 4)
GRID = task.GRID                # 8

# ------------------------------------------------------------
# Result loading helpers
# ------------------------------------------------------------
def _get_results_dir(*args) -> Path:
    module_path = args[0]
    module_dir = Path(module_path).parent
    results_dir = module_dir / "results"
    assert results_dir.is_dir(), f"No results directory under {results_dir}"
    runs = sorted(
        [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: d.name,
        reverse=True,
    )
    return runs[0]  # newest run


def _load_latest_benchmark(attempt: str) -> Dict:
    base_dir = goal_dir / "experiments" / "attention_block_2d"
    results_dir = Path(base_dir) / attempt / "results"
    bmark_path = results_dir / "benchmark.json"
    assert bmark_path.is_file(), f"benchmark.json missing at {bmark_path}"
    return json.loads(bmark_path.read_text(encoding="utf-8"))


# ------------------------------------------------------------
# Demo tab
# ------------------------------------------------------------
def _demo_content() -> gr.Blocks:
    with gr.Blocks() as demo:
        with gr.Row():
            attempt_dd = gr.Dropdown(
                label="Attempt",
                value="pass_2",
                choices=[d.name for d in (goal_dir / "experiments" / "attention_block_2d").iterdir() if d.is_dir()],
                interactive=True,
            )
            # metrics block (selectivity bar chart)
            metric_plot = gr.Plot(label="Selectivity across block sizes")
        with gr.Row():
            # heatmap for query at (0,0) for each head
            heatmap_head1 = gr.Image(label="Head b=1 attention (query tile 0x0)")
            heatmap_head2 = gr.Image(label="Head b=2 attention (query tile 0x0)")
            heatmap_head3 = gr.Image(label="Head b=4 attention (query tile 0x0)")

        def _render_demo(selected_attempt: str):
            payload = _load_latest_benchmark(selected_attempt)
            metrics = task_get_metrics(payload)  # returns a dict of metrics by key

            # Bar chart of selectivity per block size (from payload["sweep"])
            import plotly.express as px
            sel = [
                metrics.get(f"selectivity_block_{b}", 0.0)
                for b in BLOCK_SIZES
            ]
            fig = px.bar(
                x=[1, 2, 4], y=sel,
                labels={"x": "Block side b", "y": "Selectivity"},
                title="Selectivity across block sizes",
            )
            fig.update_layout(yaxis=dict(range=[0, 1]))
            metric_plot.plot(fig)

            # Load the saved attention tensor from the latest run (optional artefact)
            attn_saved_path = Path(goal_dir / "experiments" / "attention_block_2d") / selected_attempt / "results" / "run-" / "model_attn.npy"
            if attn_saved_path.is_file():
                attn = np.load(attn_saved_path.name)  # shape [B, 3, 64, 64]; take batch[0]
                # query at (0,0) is token index 0 -> tile 0,0 for every b
                # for demo we show the soft attention matrix from head 0,1,2 (each head corresponds to its own b)
                heads = [attn[0, 0], attn[0, 1], attn[0, 2]]  # [64, 64] each
                figs = []
                for h in heads:
                    import plotly.graph_objects as go
                    fig = go.Figure(data=[go.Heatmap(z=h.reshape(GRID, GRID), colorscale="RdBu_r")])
                    fig.update_layout(
                        width=200,
                        height=200,
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False, autorange="reversed"),
                    )
                    figs.append(fig)
                heatmap_head1.plot(figs[0])
                heatmap_head2.plot(figs[1])
                heatmap_head3.plot(figs[2])
            else:
                # fall back to synthetic heatmap (only used if run dir lacks the .npy artefact)
                figs = []
                for i, b in enumerate(BLOCK_SIZES):
                    att = np.zeros((GRID, GRID))
                    # query at (0,0); tile is ((0//b), (0//b))
                    for r in range(GRID):
                        for c in range(GRID):
                            if (r // b) == 0 and (c // b) == 0:
                                att[r, c] = 4.0
                            else:
                                att[r, c] = 0.1
                    att = att / att.sum()
                    import plotly.graph_objects as go
                    fig = go.Figure([go.Heatmap(z=att, colorscale="RdBu_r", opacity=0.9)])
                    fig.update_layout(
                        width=200, height=200, xaxis=dict(visible=False),
                        yaxis=dict(visible=False, autorange="reversed")
                    )
                    figs.append(fig)
                heatmap_head1.plot(figs[0])
                heatmap_head2.plot(figs[1])
                heatmap_head3.plot(figs[2])
            return metric_plot, heatmap_head1, heatmap_head2, heatmap_head3

        attempt_dd.change(
            fn=_render_demo,
            inputs=attempt_dd,
            outputs=[metric_plot, heatmap_head1, heatmap_head2, heatmap_head3],
        )
    return demo


# ------------------------------------------------------------
# Create demo with Benchmark panel
# ------------------------------------------------------------
def _benchmark_panel() -> gr.Blocks:
    panel = gr.Blocks()
    with panel:
        task.benchmark_panel(
            str(goal_dir),
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