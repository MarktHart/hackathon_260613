import gradio as gr
from agentic.experiments import benchmark_panel
from ageneric.experiments import load_task
from pathlib import Path

# Load task metadata (for panel)
task_dir = Path(__file__).parent.parent

# Benchmark panel for history across attempts
with gr.Blocks() as demo:
    with gr.Blocks() as bench:
        benchmark_panel(task_dir)  # Renders leaderboard and metric curves

    with gr.Blocks() as summary:
        gr.Markdown("# attention_sat - First Pass (hand-built model)")
        gr.Markdown(
            "This attempt computes **exact softmax attention on GPU** for the synthetic attention-saturation task.\n\n"
            "- `attn_entropy`: mean per-row entropy (lower = more peaked)\n"
            "- `attn_max`: mean max attention weight (higher = higher peak)\n"
            "- `attn_top1_frac`: mean mass in top-1 attention (higher = more concentrated)\n"
            "- `attn_topk_frac`: mean mass in top-4 attention\n"
            "- `head_labels`: ["head_0", "head_1", "head_2", "head_3"]"
        )
        gr.Markdown("The Demo tab is currently empty — this is a purely synthetic, batched experiment.\n\n"
                    "The Benchmark tab shows history across attempts with two headline metrics:\n"
                    "- `saturation_robustness_L`: how stable the top-1 fraction is across L=[16,32,64,128]\n"
                    "- `saturation_robustness_alpha`: how stable it is across Zipf exponents α=[0.0, 0.5, 1.0, 1.5]")

    demo.add_tab("Benchmark", bench)
    demo.add_tab("Summary", summary)
    # demo.add_tab("Demo", demo_demo)  # No demo needed for batched synthetic experiment


if __name__ == "__main__":
    demo.launch()