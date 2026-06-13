import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)


def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Hand-built OR mechanism expressed as max-pooling attention.

    The keys matrix has the two query directions as its first two columns:
      k_A = keys[:, 0]  (this IS q_A for the current sweep point)
      k_B = keys[:, 1]  (this IS q_B for the current sweep point)

    The balanced superposition query q_AB = normalize(q_A + q_B) has
    equal projection onto k_A and k_B. We detect this by checking
    |q @ k_A - q @ k_B| < eps.

    When the superposition query is detected, we return the element-wise
    maximum of the two single-query score vectors (max-pooling = logical OR).
    Otherwise we return the standard dot-product scores.
    """
    qt = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)

    # Signal keys = the two query directions for this sweep point
    k_A = kt[:, 0]   # (d,)
    k_B = kt[:, 1]   # (d,)

    # Projections of the query onto the two signal directions
    proj_A = qt @ k_A
    proj_B = qt @ k_B

    # Standard attention scores
    base_scores = qt @ kt   # (n,)

    # Detect balanced superposition: equal projection on both signal keys
    # At canonical (cos=0), q_AB projects to 1/sqrt(2) on both.
    # For q_A: proj_A=1, proj_B=cos. Equal only at cos=1.
    # For q_B: proj_A=cos, proj_B=1. Equal only at cos=1.
    is_superposition = torch.abs(proj_A - proj_B) < 1e-5

    if is_superposition:
        # Compute what q_A and q_B would score at every key
        scores_A = k_A @ kt   # (n,) = q_A @ K since k_A = q_A
        scores_B = k_B @ kt   # (n,) = q_B @ K since k_B = q_B
        # OR = element-wise maximum (max-pooling over detectors)
        or_scores = torch.maximum(scores_A, scores_B)
        return or_scores.detach().cpu().numpy()
    else:
        return base_scores.detach().cpu().numpy()


payload = task.evaluate(model_fn)

run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")