import gradio as gr
from agentic.experiments import load_task, benchmark_panel
import numpy as np
from matplotlib import pyplot as plt
import base64

# Canonical bin edges from task.py
DISTANCE_BIN_EDGES = np.array([0, 1, 2, 3, 4, 5, 7, 11, 17, 33, 64])
BIN_CENTERS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 9.0, 13.5, 24.5, 48.0]

def _make_head(seq_len, lambda_decay):
    attn = np.zeros((seq_len, seq_len))
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    for i in range(seq_len):
        for j in range(i + 1):
            dist = i - j
            attn[i, j] = np.exp(-dist / lambda_decay)
    attn = attn * causal_mask
    row_sums = attn.sum(axis=1, keepdims=True)
    attn = attn / (row_sums + (row_sums == 0))
    return attn

def plot_head(lambda_decay, seq_len=64):
    head = _make_head(seq_len, lambda_decay)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(head, cmap='viridis')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Attention Head (λ = {lambda_decay:.1f})')
    plt.colorbar(im, ax=ax, label='Attention Weight')
    _ = fig.canvas.draw()   # force layout and buffer
    # return plot as base64 image
    plt.close(fig)
    img_buf = fig.canvas.buffer_rgba()
    img_base64 = base64.b64encode(img_buf).decode('utf-8')
    return img_base64

def _headline_plot(model_attn, uniform_baseline):
    # Compute global mean per bin from task payload shape
    # Model_attn: list of lists lists -> shape is (4, 8, 10)
    # Uniform baseline is already a list of 10 floats.
    per_lh = np.asarray(model_attn)   # (4, 8, 10)
    global_mean = per_lh.mean(axis=(0, 1)).tolist()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(global_mean, label='Model', marker='o', linewidth=2)
    ax.plot(uniform_baseline, label='Uniform Baseline', marker='s', linewidth=2)
    ax.set_xlabel('Distance Bin Center')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('Global Distance Decay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(int(BIN_CENTERS[i])) for i in range(10)])
    _ = fig.canvas.draw()
    # return as base64
    plt.close(fig)
    img_buf = fig.canvas.buffer_rgba()
    return base64.b64encode(img_buf).decode('utf-8')

def _head_by_head_plot(model_attn):
    n_layers, n_heads, n_bins = len(model_attn), len(model_attn[0]), len(model_attn[0][0])
    per_lh = np.asarray(model_attn)   # (L, H, B) -> (4, 8, 10)
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
    axes = np.ravel(axes)
    for lh_idx, (l, h) in enumerate(zip(range(4), range(8))):
        ax = axes[lh_idx]
        head_vals = per_lh[l, h].tolist()
        ax.plot(BIN_CENTERS, head_vals, marker='x')
        ax.set_title(f'Layer {l}, Head {h}')
        ax.set_xlabel('Bin')
        ax.set_ylabel('Mean Attn')
    for ax in axes[8:]:
        ax.axis('off')
    _ = fig.canvas.draw()
    plt.close(fig)
    img_buf = fig.canvas.buffer_rgba()
    return base64.b64encode(img_buf).decode('utf-8')

def get_run_outputs():
    # In a real run, we need the latest results dir to read benchmark.json
    # For demo purposes we return a placeholder.
    return {
        "headline_plot": "<base64 placeholder>",
        "head_by_head_plot": "<base64 placeholder>"
    }

# Main Gradio Demo Block
with gr.Blocks() as demo:
    gr.Markdown("# Attention Distance Compare - Pass 2")

    # Demo Tab
    with gr.Tabs():
        with gr.Blocks():
            gr.Markdown("### Distance-Sensitive Attention Head")
            with gr.Row():
                with gr.Column():
                    lambda_param = gr.Slider(minimum=1, maximum=20, value=4, step=1, label="Distance Decay Parameter (λ)")
                    seq_len = gr.Number(value=64, label="Sequence Length")
                with gr.Column():
                    head_image = gr.Image(label="Attention Head Weight Distribution")
            def _update_head_plot(lambda_val, seq_len_val):
                img = plot_head(lambda_val, seq_len_val)
                return img
            lambda_param.change(_update_head_plot, inputs=[lambda_param, seq_len], outputs=[head_image])
            _ = lambda_param.change(_update_head_plot, inputs=[lambda_param, seq_len], outputs=[head_image])
            _ = head_image.value

        with gr.Blocks():
            gr.Label(value="Headline Summary Plot")
            with gr.Row():
                headline_plot = gr.Image()
            with gr.Blocks():
                gr.Label(value="Per Layer-Head Decay Plot")
                head_by_head_plot = gr.Image()

        with gr.Blocks():
            gr.Label(value="Benchmark Dashboard")
            with gr.Blocks():
                benchmark_panel(load_task(__file__))

    # Load demo data after block construction
    def _demo_initialize():
        try:
            # This will be populated at runtime by the pipeline once a run exists.
            demo_outputs = {
                "headline_plot": _make_headline_image(),
                "head_by_head_plot": _make_head_by_head_image()
            }
        except Exception as e:
            demo_outputs = {
                "headline_plot": "<Could not retrieve run data>",
                "head_by_head_plot": "<Could not retrieve run data>"
            }
        headline_plot.value = demo_outputs["headline_plot"]
        head_by_head_plot.value = demo_outputs["head_by_head_plot"]

    demo.load(_demo_initialize)

if __name__ == "__main__":
    demo.launch()