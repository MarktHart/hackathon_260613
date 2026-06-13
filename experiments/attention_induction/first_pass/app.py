import gradio as gr
import json
import numpy as np
from pathlib import Path
from agentic.experiments import benchmark_panel


GOAL_DIR = Path(__file__).parent.parent


def load_latest_run() -> dict:
    """Load the most recent run's payload and benchmark."""
    results_dir = GOAL_DIR / "first_pass" / "results"
    if not results_dir.exists():
        return {}

    run_dirs = sorted(results_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        return {}

    latest = run_dirs[0]
    benchmark_path = latest / "benchmark.json"
    if benchmark_path.exists():
        with open(benchmark_path) as f:
            return json.load(f)
    return {}


def create_demo_tab(benchmark_data: dict) -> gr.Blocks:
    """Create the Demo tab content."""
    if not benchmark_data:
        return gr.Markdown("No runs found. Run `main.py` first.")

    sweep = benchmark_data.get("sweep", [])
    canonical_layer = benchmark_data.get("canonical_layer", 0)
    canonical_head = benchmark_data.get("canonical_head", 0)
    n_layers = benchmark_data.get("n_layers", 12)
    n_heads = benchmark_data.get("n_heads", 12)

    # Build heatmap data: induction_score per head per layer
    heatmap = np.zeros((n_layers, n_heads))
    prev_tok_heatmap = np.zeros((n_layers, n_heads))
    random_base_heatmap = np.zeros((n_layers, n_heads))

    for rec in sweep:
        l, h = rec["layer"], rec["head"]
        heatmap[l, h] = rec["induction_score"]
        prev_tok_heatmap[l, h] = rec["prev_token_attention"]
        random_base_heatmap[l, h] = rec["random_baseline"]

    canonical_score = heatmap[canonical_layer, canonical_head]
    canonical_prev = prev_tok_heatmap[canonical_layer, canonical_head]
    canonical_random = random_base_heatmap[canonical_layer, canonical_head]

    with gr.Blocks() as demo_tab:
        gr.Markdown(f"## Induction Head Analysis — GPT-2 Small")
        gr.Markdown(
            f"**Canonical head:** Layer {canonical_layer}, Head {canonical_head} "
            f"(induction score: {canonical_score:.4f})"
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Induction Score Heatmap")
                gr.Markdown("Attention mass on the induction source position (p - rep_len + 1).")
                # Create a simple text-based heatmap since we can't easily embed matplotlib
                heatmap_text = "Layer \\ Head " + " ".join(f"{h:5d}" for h in range(n_heads)) + "\n"
                for l in range(n_layers):
                    heatmap_text += f"{l:11d} " + " ".join(f"{heatmap[l, h]:.3f}" for h in range(n_heads)) + "\n"
                gr.Code(heatmap_text, language="text", interactive=False)

            with gr.Column():
                gr.Markdown("### Previous-Token Attention (Control)")
                prev_text = "Layer \\ Head " + " ".join(f"{h:5d}" for h in range(n_heads)) + "\n"
                for l in range(n_layers):
                    prev_text += f"{l:11d} " + " ".join(f"{prev_tok_heatmap[l, h]:.3f}" for h in range(n_heads)) + "\n"
                gr.Code(prev_text, language="text", interactive=False)

            with gr.Column():
                gr.Markdown("### Random Baseline (Control)")
                rand_text = "Layer \\ Head " + " ".join(f"{h:5d}" for h in range(n_heads)) + "\n"
                for l in range(n_layers):
                    rand_text += f"{l:11d} " + " ".join(f"{random_base_heatmap[l, h]:.3f}" for h in range(n_heads)) + "\n"
                gr.Code(rand_text, language="text", interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Canonical Head Details")
        with gr.Row():
            gr.Textbox(f"Layer: {canonical_layer}", label="Layer")
            gr.Textbox(f"Head: {canonical_head}", label="Head")
            gr.Textbox(f"Induction Score: {canonical_score:.4f}", label="Induction Score")
            gr.Textbox(f"Prev Token Attention: {canonical_prev:.4f}", label="Prev Token Attention")
            gr.Textbox(f"Random Baseline: {canonical_random:.4f}", label="Random Baseline")

        # Lift over random
        lift = canonical_score - canonical_random
        selectivity = canonical_score / (canonical_score + canonical_random) if (canonical_score + canonical_random) > 1e-9 else 0.0
        gr.Markdown(f"**Lift over random:** {lift:.4f} | **Selectivity:** {selectivity:.4f}")

    return demo_tab


# Build the full app with Demo and Benchmark tabs
with gr.Blocks() as demo:
    gr.Markdown("# Attention Induction — First Pass (GPT-2 Small)")

    with gr.Tabs():
        with gr.TabItem("Demo"):
            benchmark_data = load_latest_run()
            demo_tab = create_demo_tab(benchmark_data)
            demo_tab.render()

        with gr.TabItem("Benchmark"):
            # This loads the shared benchmark panel across all attempts
            benchmark_panel(GOAL_DIR).render()

if __name__ == "__main__":
    demo.launch()