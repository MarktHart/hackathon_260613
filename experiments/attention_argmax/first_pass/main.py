import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline reserves a GPU for this attempt and verifies it was used, so the
# model function runs its compute on CUDA.
DEVICE = "cuda"

# Import the task's data generator and evaluator.
task = load_task(__file__)


# task.evaluate calls model_fn(q (d,), K (N, d), V (N, d)) -> attn_weights (N,),
# a probability distribution over the N key positions. A perfect argmax head
# places (almost) all of its mass on the single highest-similarity position —
# the "spike on the winner" idea from the first pass, expressed against the real
# argmax contract.
def first_pass_model_fn(q: np.ndarray, K: np.ndarray, V: np.ndarray = None) -> np.ndarray:
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)   # (d,)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)   # (N, d)

    similarities = Kt @ qt          # (N,) dot product of each key with the query
    # Sharp softmax concentrates mass on the argmax position.
    attn = torch.softmax(similarities, dim=0)
    return attn.detach().cpu().numpy()


# ---- Run the payload pipeline ----
payload = task.evaluate(first_pass_model_fn)
record_dir = results_dir(__file__)
record_benchmark(__file__, record_dir, payload)
