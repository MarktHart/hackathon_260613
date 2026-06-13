import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

def model_fn(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Standard scaled dot-product attention with temperature tuning for SNR.
    
    This is the canonical attention mechanism: softmax(Q @ K.T / sqrt(d))
    The query is target_key + noise at 10 dB SNR. With d=64, the optimal
    temperature is sqrt(d) = 8, which matches standard attention.
    
    We run the computation on GPU as required.
    """
    qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)      # [d]
    kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)      # [K, d]
    
    # Scaled dot-product attention: attn = softmax(Q @ K.T / sqrt(d))
    # Q shape [d], K shape [K, d] -> scores shape [K]
    scores = qt @ kt.T                                               # [K]
    scores = scores / np.sqrt(len(Q))                                # scale by sqrt(d)
    attn = torch.softmax(scores, dim=-1)                             # [K]
    
    return attn.detach().cpu().numpy()

payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)