import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees at least one GPU.
DEVICE = "cuda"

def model_fn(coords: np.ndarray, current_idx: int, visited: np.ndarray) -> np.ndarray:
    """
    Hand-coded attention mechanism for nearest-neighbor TSP routing.
    
    Implements attention as: logits = -||current_pos - city_pos||^2
    This is equivalent to a dot-product attention where Q=current_city, K=all_cities,
    with a learned (hand-coded) linear projection that extracts the negative squared distance.
    """
    n = coords.shape[0]
    
    # Move to GPU as torch tensors
    coords_t = torch.as_tensor(coords, dtype=torch.float32, device=DEVICE)  # (n, 2)
    current_t = coords_t[current_idx:current_idx+1]  # (1, 2)
    visited_t = torch.as_tensor(visited, dtype=torch.bool, device=DEVICE)  # (n,)
    
    # Attention computation: negative squared Euclidean distance
    # diff shape: (n, 2)
    diff = current_t - coords_t
    # sqdist shape: (n,)
    sqdist = (diff * diff).sum(dim=-1)
    
    # Logits = negative distance (so closer = higher logit)
    # Scale by a temperature factor to sharpen attention
    temperature = 10.0
    logits = -sqdist * temperature
    
    # Mask visited cities with large negative value (handled by evaluator too,
    # but we do it here for completeness in the attention mechanism)
    logits = logits.masked_fill(visited_t, -1e9)
    
    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    print("Running pass_2 attention-based NN router on CUDA...")
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    print("Payload keys:", list(payload.keys()))
    for rec in payload["sweep"]:
        print(f"  n={rec['n']}: nn_acc={rec['nn_accuracy']:.4f}, tour_ratio={rec['tour_length_ratio']:.4f}")
    record_benchmark(__file__, run_dir, payload)
    print("Results written to", run_dir)
