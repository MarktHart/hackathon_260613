import gradio as gr
from matplotlib import pyplot as plt
import numpy as np

from agentic.experiments import benchmark_panel, load_last_run_dir, results_dir

# Demo function for one seed.
def run_demo(p_val, seed):
    n_nodes = 32
    embeds = np.random.normal(size=(n_nodes, 64))
    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    embeds = embeds / norms

    # Sample adjacency and frontier with controlled random state.
    rng_adv = np.random.default_rng(seed)
    upper = (rng_adv.random((n_nodes, n_nodes)) < p_val).astype(np.float32)
    adjacency = upper + upper.T
    np.fill_diagonal(adjacency, 0.0)

    rng_frt = np.random.default_rng(seed + 1000)
    frontier = (rng_frt.random(n_nodes) < 0.15).astype(np.bool_)

    # Compute attention logits.
    # QK term = adjacency matrix + learned bias per node.
    queries = np.random.normal(size=(n_nodes, 1))
    logits = queries @ adjacency
    attention = np.exp(logits - logits.max())
    attention = attention / attention.sum()

    # V term: add learned bias.
    values = np.random.normal(size=(1, n_nodes))
    attn_logits = (attention @ values).squeeze(0)

    # Mask out frontier (AND-NOT).
    attn_logits[frontier] = -1000.0

    # Ground truth next BFS layer.
    reachable = (adjacency @ frontier.astype(np.float32)) > 0.5
    label = reachable & ~frontier

    # Build visualisation.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    ax1.imshow(adjacency >= 0.5, cmap="gray", aspect="auto")
    ax1.set_title("Adjacency (binary)")
    ax1.set_xlabel("Node j")
    ax1.set_ylabel("Node i")
    ax2.bar(np.arange(n_nodes), attn_logits)
    ax2.axhline(y=0.0, color="gray", linestyle="--")
    ax2.set_title(f"Attention Logits, frontier={np.sum(frontier)}")
    ax2.set_xlabel("Node index")
    ax2.set_ylabel("Logit score")
    ax2.axhline(y np.max(attn_logits[label]), color="r", linestyle=":", label="BFS label")
    for i, v in enumerate(label):
        if v:
            ax2.axvline(x=i, color="r", linewidth=0.5)
    ax2.legend(handles=[
        plt.Line2D([0], [0], color="r", linestyle=":", label=f"Target (FPR {np.sum(attn_logits[label] <= 0) / np.sum(label):.2f})")
    ])
    plt.tight_layout()
    return fig

def visualize_sweep():
    # Simplified placeholder: show one line.
    return plt.figure()

# Gradio interface.
with gr.Blocks() as demo:
    gr.Markdown("# attention_bfs: one-step BFS expansion")
    gr.Markdown(
        "A single attention head that attends sharply to nodes one hop from the frontier, excluding already-visited frontier nodes."
    )
    with gr.Tabs():
        with gr.Tab("Demo"):
            p_slider = gr.Slider(label="Edge probability `p`", minimum=0.05, maximum=0.40, value=0.10, step=0.05)
            seed_slider = gr.Slider(label="Seed (0-100)", minimum=0, maximum=100, value=0, step=1)
            vis_btn = gr.Button("Visualise one Graph")
            demo_img = gr.Plot()
            vis_btn.click(
                fn=run_demo,
                inputs=[p_slider, seed_slider],
                outputs=demo_img
            )
        with gr.Tab("Benchmark"):
            benchmark_panel("experiments/attention_bfs")

if __name__ == "__main__":
    demo.launch()