import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees at least one GPU.
DEVICE = "cuda"

def model_fn(coords: np.ndarray, current_idx: int, visited: np.ndarray) -> np.ndarray:
    n = int(coords.shape[0])
    qt = torch.as_tensor(coords, dtype=torch.float32, device=DEVICE)
    cur = torch.as_tensor(coords[current_idx], dtype=torch.float32, device=DEVICE).unsqueeze(0)
    diff = cur - qt
    # Squared Euclidean distance (sqrt is monotonic so can skip)
    sqdist = (diff * diff).sum(dim=-1)  # (n,)
    # Inverse distance = proximity signal
    proximity = 1.0 / (sqdist + 1e-8)   # avoid zero division
    # Scale roughly to a plausible logit range
    proximity = proximity * 10.0

    # Return logits on CPU per `task.py` contract
    return proximity.detach().cpu().numpy()


if __name__ == "__main__":
    print("Running first_pass model on CUDA...")
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    print("Payload keys:", list(payload.keys()))
    record_benchmark(__file__, run_dir, payload)
    print("Results written to", run_dir)