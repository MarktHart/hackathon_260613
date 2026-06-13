import gradio as gr
import json
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent

def _load_latest_run():
    results_dir = GOAL_DIR / "pass_3" / "results"
    if not results_dir.exists():
        return {}
    latest = sorted(results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)[0]
    benchmark_path = latest / "benchmark.json"
    if not benchmark_path.exists():
        return {}
    with open(benchmark_path) as f:
        return json.load(f)


def _heatmap_text(arr, title="Heatmap", width=14):
    ncols = arr.shape[1]
    head_str = f"{title} | {''.join(f'{i:3d}' for i in range(ncols))}\n"
    rows = [" " * width + "-" * (width * ncols), " |" + "-" * (width * ncols * 2)]
    for i, row in enumerate(arr):
        rows.append(f"{i:5d}{(' ').join(f' |{v:.3f}' for v in row)}")
    return "\n".join(rows)


def create_demo_tab(benchmark_data: dict):
    if not benchmark_data:
        return gr.Markdown("⚠️ No runs found. Run `main.py` first to train the induction model.")

    sweep = benchmark_data.get("sweep", [])
    cano_l = benchmark_data.get("canonical_layer", 1)   # induction head lives at layer 1
    cano_h = benchmark_data.get("canonical_head", 0)   # head 0

    # Build a heat‑map of induction score (accuracy) across layer × head
    max_dist = 64
    n_layers = 3
    n_heads = 4
    heatmap = np.zeros((n_layers, n_heads), dtype=float)
    for rec in sweep:
        layer, head = rec["layer"], rec["head"]
        if layer in [0, 1, 2] and head in [0, 1, 2, 3]:   # our model only has 4 heads
            heatmap[layer, head] = rec.get("accuracy", 0.0)

    # Heat‑map text for demo (text block)
    heatmap_text = _heatmap_text(arr=heatmap, title="Induction Accuracy (Layer × Head)", width=14)

    lift = float(benchmark_data.get("lift_over_uniform", 0))
    selectivity = float(benchmark_data.get("induction_selectivity", 0))

    with gr.Blocks() as demo:
        gr.Markdown(f"## Minimal Induction Demonstration — 3‑layer model (vocab=128, seq_len=192)")
        gr.Markdown(f"**Headline:** canonical head L={cano_l} H={cano_h} "
                    f"has accuracy `{heatmap[cano_l, cano_h]:.3f}`\n"
                    "**Selectivity:** `{selectivity:.3f}` | **Lift over uniform:** `{lift:.3f}`")

        with gr.Row():
            gr.Code(label="Induction Accuracy Heatmap (Layer × Head)",
                    code=heatmap_text, language="text")
            gr.Markdown(
                f"Layer‑0 is generic context; layer‑1 contains the induction head; layer‑2 is another context attention block."
            )
        with gr.Row():
            gr.Label("Induction head location", value=f"Layer {cano_l}, Head {cano_h}")

    return demo


# Build the full app with Demo and Benchmark tabs
with gr.Blocks() as demo:
    gr.Markdown("# Attention Induction — pass_3: small hand‑coded induction head")

    demo_tabs = gr.Tabs()
    with demo_tabs:
        with gr.TabItem("Demo"):
            benchmark_data = _load_latest_run()
            if benchmark_data and "sweep" in benchmark_data:
                demo_tab = create_demo_tab(benchmark_data)
                demo_tab.render()
            else:
                gr.Markdown("No runs found. Train the model with `main.py` first.")
        with gr.TabItem("Benchmark"):
            # Pull in the shared leaderboard and plot dashboard
            benchmark_panel(GOAL_DIR).render()


if __name__ == "__main__":
    demo.launch()