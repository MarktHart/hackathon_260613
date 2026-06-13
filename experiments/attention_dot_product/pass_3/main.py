import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"


def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    scaled dot-product attention on the GPU.

    Q, K, V : (batch, n_heads, seq_len, d_head)
    Returns: (batch, n_heads, seq_len, d_head)
    """
    Qt = torch.from_numpy(Q).float().to(DEVICE)
    Kt = torch.from_numpy(K).float().to(DEVICE)
    Vt = torch.from_numpy(V).float().to(DEVICE)

    d_head = Qt.shape[-1]
    scale = 1.0 / torch.sqrt(torch.tensor(d_head, device=DEVICE))
    scores = torch.einsum("bhsd,bhtd->bhst", Qt, Kt) * scale   # (B, H, S, S)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhst,bhtd->bhsd", attn, Vt)            # (B, H, S, d_head)
    return out.detach().cpu().numpy()


payload = task.evaluate(model_fn)
record_benchmark(__file__, results_dir(__file__), payload)
