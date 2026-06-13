import gradio as gr
import json
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel

GOAL_DIR = Path(__file__).parent.parent

def _load_latest_run():
    results_dir = GOAL_DIR / "pass_2" / "results"
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
    cano_l = benchmark_data.get("canonical_layer", 0)
    cano_h = benchmark_data.get("canonical_head", 0)
    n_layers, n_heads = 3, 4  # our model's size (hardcoded for this attempt)

    # Build a heat‑map of induction score across layer × head
    heatmap = np.zeros((n_layers, n_heads))
    for rec in sweep:
        l, h = rec["layer"], rec["head"]
        heatmap[l, h] = rec["induction_score"]

    # Heat‑map text (text block) for demo purposes (Gradio blocks code)
    heatmap_text = _heatmap_text(arr=heatmap, title="Induction Score")

    lift = float(benchmark_data.get("lift_over_baseline_canonical", 0))
    selectivity = float(benchmark_data.get("induction_selectivity", 0))

    with gr.Blocks() as demo_tab:
        gr.Markdown(f"## Minimal Induction Demonstration — 3‑layer model (vocab=100, seq_len=64)")
        gr.Markdown(f"**Headline:** canonical head L={cano_l} H={cano_h} "
                    f"has induction score `{heatmap[cano_l, cano_h]:.3f}`\n"
                    "**Selectivity:** `{selectivity:.3f}` | **Lift over baseline:** `{lift:.3f}`")

        with gr.Row():
            gr.Code(label="Induction Score Heatmap (Layer × Head)", code=heatmap_text, language="text")
            gr.Markdown(
                f"Layer‑0 is generic context attention; layer‑1 contains the induction head; layer‑2 is another attention block."
            )
        with gr.Row():
            gr.Label("Induction head location", value=f"Layer {cano_l}, Head {cano_h}")

    return demo_tab

# Build the full app with Demo and Benchmark tabs
with gr.Blocks() as demo:
    gr.Markdown("# Attention Induction — pass_2: small trained model")

    demo_tabs = gr.Tabs()
    with demo_tabs:
        with gr.TabItem("Demo"):
            benchmark_data = _load_latest_run()
            if benchmark_data:
                demo_tab = create_demo_tab(benchmark_data)
                demo_tab.render()
            else:
                gr.Markdown("No runs found. Train the model with `main.py` first.")
        with gr.TabItem("Benchmark"):
            # This loads the shared benchmark panel across all attempts
            benchmark_panel(GOAL_DIR).render()

if __name__ == "__main__":
    demo.launch()