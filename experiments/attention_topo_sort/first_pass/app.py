import torch
import gradio as gr
from pathlib import Path
import json

from agentic.experiments import (
    benchmark_panel,
    list_runs,
    load_task,
    load_payload,
    ResultsDir,
)

import numpy as np
from collections import defaultdict

# --------------------------- Demo Panel ---------------------------------
def demo_panel():
    with gr.Blocks() as demo:
        gr.Markdown("# Attention Topo Sort Demo")
        gr.Markdown("Visualising headwise topological consistency for the bracket-ordered synthetic task.")

        with gr.Tabs():
            with gr.Tab("Head Heatmaps & Summary"):
                with gr.Blocks():
                    head_selector = gr.Dropdown(choices=[f"Head {i}" for i in range(12)], label="Select Head")
                    head_selector.change(
                        fn=_update_head_heatmap,
                        inputs=head_selector,
                        outputs=[heatmap_fig, heatmap_caption, stats_panel],
                    )

                    with gr.Row():
                        with gr.Column():
                            heatmap_caption = gr.Markdown()
                        with gr.Column():
                            stats_panel = gr.Markdown()

            with gr.Tab("All Heads Overview"):
                overview_btn = gr.Button("Show All Head Consistencies")
                overview_btn.click(
                    fn=_render_overview,
                    outputs=overview_panel,
                )

                overview_panel = gr.Markdown()

    if __name__ == "__main__":
        demo.launch()
    return demo


# Helper to render a 12x1 head consistency bar chart.
def _render_overview() -> str:
    run_dir = _get_latest_run()
    payload_path = Path(run_dir) / "benchmark.json"
    with open(payload_path) as f:
        payload = json.load(f)

    h = payload["head_metrics"]
    c = [hm["topo_consistency"] for hm in h]

    rows = [
        f"Head {i}: {c[i]:.3f}"
        for i in range(len(c))
    ]
    return "\n".join(rows)


# Helper to update the head heatmap view.
def _update_head_heatmap(head_idx_str: str) -> tuple[str, str, str]:
    h_idx = int(head_idx_str.split(" ")[1])
    run_dir = _get_latest_run()
    payload_path = Path(run_dir) / "benchmark.json"
    with open(payload_path) as f:
        payload = json.load(f)

    batch_attn = np.load(Path(run_dir) / "batch_attn.npy")
    batch_ids = payload["canonical_batch"]["input_ids"]
    batch_mask = payload["canonical_batch"]["attention_mask"]

    b = batch_attn[0, h_idx]   # first example, selected head
    fig_data = _heatmap_data(b, batch_ids, batch_mask)
    return fig_data, f"Head {h_idx} attention (topological ancestors highlighted)", _ stats_for_head(h_idx, payload["head_metrics"])


# Internal utilities.
def _stats_for_head(head_idx: int, head_metrics: list[dict]) -> str:
    hm = head_metrics[head_idx]
    return (
        f"Topo consistency: {hm.get('topo_consistency', 0.0):.3f}\n"
        f"Opening bracket consistency: {hm.get('topo_consistency_opening', 0.0):.3f}\n"
        f"Closing bracket consistency: {hm.get('topo_consistency_closing', 0.0):.3f}\n"
        f"Mass to ancestors: {hm.get('mass_to_ancestors', 0.0):.3f}\n"
        f"Mass to non-ancestors: {hm.get('mass_to_non_ancestors', 0.0):.3f}\n"
        f"Entropy: {hm.get('entropy', 0.0):.2f} nats"
    )


def _heatmap_data(attn: np.ndarray, tokens: np.ndarray, mask: np.ndarray) -> str:
    # Build a simple textual heatmap: show the token index at each key position, and an attention percentage at each cell.
    seq_len = attn.shape[0]
    heatmap_markdown = ""
    header = " ".join(f"{i}({t})" for i, t in enumerate(tokens) if tokens[i] != 0)
    heatmap_markdown += f"Query \\ Key   {header}\n"

    for q in range(seq_len):
        if not mask[q]:
            continue
        row = [f"{attn[q][k]:.2f}" if mask[k] else "0.00" for k in range(seq_len)]
        heatmap_markdown += f"{q}({tokens[q]}): {' '.join(row)}\n"
    return heatmap_markdown


def _get_latest_run() -> str:
    run_dirs = sorted(Path("results").rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(run_dirs[0]) if run_dirs else "."


# --------------------------- Benchmark Overview Panel -------------------------------
def _render_benchmark_panel(goal_dir: str) -> str:
    # Reuse the standard panel to show the leaderboard and metric history.
    return benchmark_panel(goal_dir)


# --------------------------- Gradio Demo Hook -------------------------------
demo = demo_panel()

if __name__ == "__main__":
    demo.launch()