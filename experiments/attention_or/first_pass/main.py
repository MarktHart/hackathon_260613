import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)


def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Hand-built OR attention head.

    Contract (from task.evaluate): query is a (d,) unit vector, keys is the
    (d, n) key matrix whose columns 0 and 1 are the signal keys k_A, k_B and
    whose remaining columns are random noise keys. Return a (n,) score vector,
    one attention score per key.

    The mechanism is a plain dot-product attention score: score_j = <query, k_j>.
    With the balanced-superposition query q_AB = normalize(q_A + q_B), a single
    head fires on *both* signal keys at once — i.e. it implements OR.
    """
    q_t = torch.as_tensor(np.asarray(query), dtype=torch.float32, device=DEVICE)
    K_t = torch.as_tensor(np.asarray(keys), dtype=torch.float32, device=DEVICE)
    if q_t.ndim != 1:
        raise ValueError(f"model_fn expects a 1-D query (d,), got shape {q_t.shape}")
    # keys: (d, n); query: (d,) -> scores: (n,)
    scores = K_t.t() @ q_t
    return scores.detach().cpu().numpy()


payload = task.evaluate(model_fn)

run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")
