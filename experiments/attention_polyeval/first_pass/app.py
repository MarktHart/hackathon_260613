import gradio as gr
from agentic.experiments import load_task, benchmark_panel
import numpy as np

def model_fn(x_ctx: np.ndarray, y_ctx: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    # Same hand-built model as main.py, copied here for the notebook demo.
    import torch
    DEVICE = "cuda"

    N_CTX = x_ctx.shape[1]
    B = x_ctx.shape[0]
    N_QUERY = x_query.shape[1]

    x_ctx_t = torch.as_tensor(x_ctx, dtype=torch.float32, device=DEVICE)
    y_ctx_t = torch.as_tensor(y_ctx, dtype=torch.float32, device=DEVICE)
    x_query_t = torch.as_tensor(x_query, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        # Hardcoded learnable weights from main.py
        Q_w = torch.tensor([[-0.9716, -0.2942, -0.4856, 0.3272, -0.9721, 1.1437, -0.6107, -0.0918,
                              -0.8449,  0.4514,  0.4671, 0.7879,  0.4449, -0.6812,  0.6358,  1.0034],
                            [0.2776, -1.2917, -0.2780, 0.3758, -0.7957, -0.2441, -0.3566,  0.3903,
                             0.1069, 0.5455,  1.5811, -0.5345, -0.4659,  0.2611,  0.3452, -0.1977]],
                           device=DEVICE, dtype=torch.float32)
        K_w = torch.tensor([[-0.3151, -0.4030, -0.3877, -0.1748,  0.0372, -0.8147,  1.2577, -0.3033,
                              -0.3093, -1.0460, -0.3594, -0.4280,  0.2408,  0.0638, -0.0490, -0.1850],
                            [0.2147, -0.2252,  0.1782,  0.7326,  1.8170, -0.0365, -0.0781, -0.1189,
                             0.0862,  1.4651,  1.4917, -0.4894,  0.9436,  0.6299,  1.2865, -0.0986]],
                           device=DEVICE, dtype=torch.float32)
        bias = torch.tensor([0.2855, -1.4826, -0.3483, 0.0771, -0.0422, 0.1747, -1.6280,  0.4874,
                              0.9834, -0.1449, -0.8480, -0.1762,  0.9873, -0.3035,  0.1628,  0.5204],
                           device=DEVICE, dtype=torch.float32)

    x_all = torch.cat([x_ctx_t.unsqueeze(-1), y_ctx_t.unsqueeze(-1)], dim=-1)  # (B, 1, 2 * N_CTX)
    Q = torch.matmul(x_all, Q_w)  # (B, 1, 16)
    K = torch.matmul(x_all, K_w)  # (B, 1, 16)
    attn_weights = (Q @ K.transpose(-1, -2)) * (128.0 ** 0.5)
    attn_out = attn_weights.sum(-1, keepdim=True)  # (B, 1, 1)  # Sum over the small head dimension
    y_pred = attn_out + bias.unsqueeze(-1).unsqueeze(-2)  # broadcast (B, 1, 16)
    # But we want a single output per episode; use the first value of the head
    y_pred = y_pred[:, 0, 0:1]  # (B, 1)
    y_pred = y_pred.expand(B, N_QUERY)  # (B, N_QUERY)

    return y_pred.detach().cpu().numpy()

def demo_panel():
    with gr.Blocks() as demo:
        with gr.Tab("Demo"):
            gr.Markdown("Demonstration notebook")
            with gr.Blocks():
                with gr.Row():
                    ctx_input = gr.NumpyInput(shape=(100, 32), label="x_context (2d)")
                    y_input = gr.NumpyInput(shape=(100, 32), label="y_context (2d)")
                    query_input = gr.NumpyInput(shape=(100, 16), label="x_query (2d)")
                with gr.Row():
                    btn = gr.Button("Predict")
                with gr.Row():
                    output = gr.NumpyOutput()
            btn.click(model_fn, inputs=[ctx_input, y_input, query_input], outputs=output)

        with gr.Tab("Benchmark History"):
            panel = benchmark_panel("experiments/attention_polyeval")
            panel.render()

    return demo

with gr.Blocks() as demo:
    panel = demo_panel()

if __name__ == "__main__":
    demo.launch()