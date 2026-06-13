import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# Import the task's data generator and evaluator
task = load_task(__file__)


# task.evaluate calls model_fn(q_A (d,), q_B (d,), residual (n_positions, d))
# and expects per-position logits (n_positions,). The AND mechanism fires only
# where BOTH feature directions are present, realised as the product of the two
# probe projections (a conjunction): high only when residual aligns with q_A AND q_B.
def model_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
    q_At = torch.as_tensor(q_A, dtype=torch.float32, device=DEVICE)          # (d,)
    q_Bt = torch.as_tensor(q_B, dtype=torch.float32, device=DEVICE)          # (d,)
    rt = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)       # (n, d)

    a = rt @ q_At          # (n,) projection onto feature A
    b = rt @ q_Bt          # (n,) projection onto feature B
    logits = a * b         # conjunction: large only when both are present
    return logits.detach().cpu().numpy()


# Run the task on the model function
payload = task.evaluate(model_fn)

# Record the benchmark results
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
