import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"


def model_fn(query: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
    """
    Hand-built one-hot attention mechanism.

    The synthetic task constructs keys such that:
    - Target key = query (unit vector)
    - All other keys are orthogonal to query

    Therefore, dot-product attention scores = keys @ query gives:
    - Score at target position = 1.0
    - Scores at noise positions = 0.0

    With temperature τ=0.1, softmax(10, 0, 0, ...) concentrates ~99% mass on target.
    This required no training — the mechanism follows directly from the task structure.
    """
    qt = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)

    # Dot-product attention scores: (L,)
    scores = kt @ qt
    scores = scores / temperature

    attn = torch.softmax(scores, dim=-1)

    return attn.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)


if __name__ == "__main__":
    main()