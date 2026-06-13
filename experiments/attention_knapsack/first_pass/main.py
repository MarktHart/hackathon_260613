import sys
import torch
from pathlib import Path

from agentic.experiments import load_task, record_benchmark, results_dir
from agentic.experiments.base_model import BaseAttentionModel
import numpy as np

 DEVICE = "cuda"  # pipeline guarantees a GPU

def attention_head_scores(
    item_values: np.ndarray,   # [batch, n_items]
    item_weights: np.ndarray,  # [batch, n_items]
    capacity: np.ndarray       # [batch]
) -> np.ndarray:               # [batch, n_items] — selection scores
    """
    Hand-built attention circuit that scores each item by its (value / weight)
    ratio adjusted for how much capacity would be left after picking it.
    """
    batch, n_items = item_values.shape
    scores = np.zeros((batch, n_items), dtype=np.float32)

    for b in range(batch):
        values = item_values[b]
        weights = item_weights[b]
        cap = capacity[b]

        # Value-per-unit-weight (greedy ratio baseline)
        ratios = values / (weights + 1e-9)

        # Discount each item's score by the fraction of capacity it consumes
        # relative to the total, to encourage smaller items when capacity is tight.
        total_w = weights.sum()
        frac_w = weights / (total_w + 1e-9)
        # Score = ratio * (1 - frac_w) — higher ratio, lower share -> higher priority
        scores[b] = ratios * (1.0 - frac_w)

    return scores


def main() -> None:
    # Load task and canonical batch (CPU/NumPy only)
    task = load_task(__file__)
    batch = task.generate(seed=0)

    # Define the GPU-executing model function
    def my_model_fn(item_values, item_weights, capacity):
        # Convert to GPU for the heavy compute
        item_v_t = torch.as_tensor(item_values, dtype=torch.float32, device=DEVICE)
        item_w_t = torch.as_tensor(item_weights, dtype=torch.float32, device=DEVICE)
        capacity_t = torch.as_tensor(capacity, dtype=torch.float32, device=DEVICE)

        batch, n_items = item_v_t.shape
        scores_t = torch.zeros((batch, n_items), dtype=torch.float32, device=DEVICE)

        for b in range(batch):
            values = item_v_t[b]
            weights = item_w_t[b]
            cap = capacity_t[b]

            ratios = values / (weights + 1e-9)

            total_w = weights.sum()
            frac_w = weights / (total_w + 1e-9)
            scores_t[b] = ratios * (1.0 - frac_w)

        return scores_t.detach().cpu().numpy()

    # Evaluate
    payload = task.evaluate(my_model_fn)

    # Record
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Wrote benchmark results to {run_dir} / benchmark.json")


if __name__ == "__main__":
    main()