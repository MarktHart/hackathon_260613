import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# The signature is hard-coded from task.py:
#   def model_fn(tokens: np.ndarray) -> np.ndarray:
#       # tokens : (N, 4) int32
#       # returns: (N,) float array of logits; XOR=1 iff logit > 0
def model_fn(tokens: np.ndarray) -> np.ndarray:
    tok = torch.as_tensor(tokens, device=DEVICE)
    # Decode A = tokens[:,1] ∈ {1,2} → 0,1
    A = (tok[:, 1] - 1).to(torch.float32)
    # Decode B = tokens[:,2] ∈ {3,4} → 0,1
    B = (tok[:, 2] - 3).to(torch.float32)
    # Compute XOR as 1 - (A == B) using the squared difference.
    # This is the minimal quadratic circuit over superposition:
    #   (A - B)^2 = 0 when A == B
    #       = 1 when A != B
    # The logit is exactly this XOR signal.
    logits = (A - B) ** 2  # shape (N,)
    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    task = load_task(__file__)
    # `evaluate` runs model_fn across every sweep point,
    # building a dict that benchmark.py can score.
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)