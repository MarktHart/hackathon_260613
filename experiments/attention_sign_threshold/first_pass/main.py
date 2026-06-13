import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

# Hand-built sharpened attention: scale cosine by temperature and apply tanh
# to create a sharp but smooth sign flip at tau=0. beta controls sharpness.
BETA = 20.0  # temperature: larger = sharper threshold


def model_fn(q: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Pre-softmax attention score with sharp sign threshold at cos(q,k)=0.

    Computes cosine similarity (q·k since inputs are unit-norm), applies
    a high-gain tanh to sharpen the transition through zero.
    """
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(k, dtype=torch.float32, device=DEVICE)
    cosine = torch.sum(qt * kt, dim=1)  # (n_pairs,)
    scores = torch.tanh(BETA * cosine)
    return scores.detach().cpu().numpy()


payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")