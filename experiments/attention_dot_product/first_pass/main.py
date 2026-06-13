import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"


def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    # Q, K, V : (batch, n_heads, seq_len, d_head)
    # Attention output = softmax(QK^T / sqrt(d_head)) @ V
    # --- GPU compute (torch on CUDA) ---
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    Vt = torch.as_tensor(V, dtype=torch.float32, device=DEVICE)

    d_head = Qt.shape[-1]
    scale = 1.0 / (d_head ** 0.5)
    scores = torch.einsum("bhsd,bhtd->bhst", Qt, Kt) * scale  # (B, H, S, S)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", attn, Vt)           # (B, H, S, d_head)
    return out.detach().cpu().numpy().astype(np.float32)


payload = task.evaluate(model_fn)
record_benchmark(__file__, results_dir(__file__), payload)