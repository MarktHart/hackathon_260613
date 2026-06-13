import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# Pipeline guarantees a GPU
DEVICE = "cuda"

# Task imports
task = load_task(__file__)

def model_fn(x_ctx: np.ndarray, y_ctx: np.ndarray, x_query: np.ndarray) -> np.ndarray:
    """Hand-built attention circuit for in-context polynomial regression.

    x_context: (batch, n_ctx)
    y_context: (batch, n_ctx)
    x_query:   (batch, n_query)
    returns:   (batch, n_query)

    Circuit:
      - Concatenate x/y per token => token vector of shape (batch, 1, 2 * n_ctx)
      - Project to Q and K with learnable weights and bias -> (batch, 1, n_ctx)
      - Scaled dot-product attention (no softmax denominator) -> (batch, 1, n_ctx)
      - Sum along context dim and add learned bias -> (batch, n_query)
    """

    N_CTX = x_ctx.shape[1]
    B = x_ctx.shape[0]
    N_QUERY = x_query.shape[1]

    # Convert NumPy to GPU tensor
    x_ctx_t = torch.as_tensor(x_ctx, dtype=torch.float32, device=DEVICE)
    y_ctx_t = torch.as_tensor(y_ctx, dtype=torch.float32, device=DEVICE)
    x_query_t = torch.as_tensor(x_query, dtype=torch.float32, device=DEVICE)

    # Learnable parameters
    # Q and K projections share no weights; each is a (2, d_head) matrix.
    # These are initialized from N(0, 0.02) (standard GPT-2 init for small weights)
    # and will be used as-is in this first-pass synthetic attempt.
    with torch.no_grad():
        Q_w = torch.normal(0.0, 0.02, (2, 16), device=DEVICE)   # (2, 16)
        K_w = torch.normal(0.0, 0.02, (2, 16), device=DEVICE)   # (2, 16)
        bias = torch.normal(0.0, 1.0, (16,), device=DEVICE)   # (16,)

    # Flatten context into a sequence of (x, y) values -> (B, 1, 2 * N_CTX)
    x_all = torch.cat([x_ctx_t.unsqueeze(-1), y_ctx_t.unsqueeze(-1)], dim=-1)
    # Reshape into (B, 1, 2 * N_CTX)
    x_all = x_all.view(B, 1, 2 * N_CTX)

    # Project to Q and K (each context token gets its own Q and K vector)
    # Q: (B, 1, 2 * N_CTX) -> (B, 1, N_CTX) after matmul
    Q = torch.matmul(x_all, Q_w)   # (B, 1, 16)
    # K: same projection
    K = torch.matmul(x_all, K_w)   # (B, 1, 16)

    # Scaled dot-product attention
    # (B, 1, 16) @ (B, 1, 16).T -> (B, 1, N_CTX)
    attn_weights = (Q @ K.transpose(-1, -2)) * (128.0 ** 0.5)
    # For polynomial regression, we don't need softmax; we just sum
    attn_out = attn_weights.sum(-1, keepdim=True)   # (B, 1, 1)

    # Add learned bias (same for all queries)
    y_pred = attn_out + bias.unsqueeze(0)   # (B, 1, 16) after broadcast? Actually just scalar

    # Expand to (B, N_QUERY) by copying the constant prediction across all queries
    y_pred = y_pred.expand(B, N_QUERY)

    return y_pred.detach().cpu().numpy()

# Run the evaluation
payload = task.evaluate(model_fn)

# Record benchmark
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)