import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)

def model_fn(input_ids: np.ndarray, start: int, end: int) -> float:
    # Hand-built baseline: compute the exact range sum on the GPU using a cumsum
    # prefix difference. Mirrors the generator's own method (see task.generate)
    # so it scores zero mse at every k, proving generator and scorer agree.
    seq = torch.as_tensor(input_ids, dtype=torch.float32, device=DEVICE)
    cumsum = torch.cat([torch.zeros(1, dtype=torch.float32, device=DEVICE),
                        torch.cumsum(seq, dim=0)], dim=0)  # (L+1,)
    total = cumsum[int(end)] - cumsum[int(start)]
    return float(total.item())

payload = task.evaluate(model_fn)
record_benchmark(__file__, results_dir(__file__), payload)
