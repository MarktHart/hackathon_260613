import torch
import numpy as np

from agentic.experiments import load_task, record_benchmark, results_dir

# GPU guarantee; no fallback.
DEVICE = "cuda"


def model_fn(adjacency: np.ndarray, source: int, hops: int) -> np.ndarray:
    """
    Iterative attention-based BFS propagation.

    Each step treats the current frontier as a query distribution attending to
    neighbors via the adjacency matrix (key/value). This is exactly the
    operation of one attention head where Q = frontier, K = V = adjacency.
    Stacking `hops` such steps implements multi-hop reachability.
    """
    n = adjacency.shape[0]
    adj_t = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)

    # Frontier starts as one-hot at the source node.
    frontier = torch.zeros(n, device=DEVICE)
    frontier[source] = 1.0

    # Visited accumulates all nodes reached so far (source + all hops).
    visited = frontier.clone()

    for _ in range(hops):
        # Attention step: frontier (query) attends to adjacency (key/value).
        # Scores = frontier @ adj_t  →  (n,) giving probability mass passed to each node.
        scores = frontier @ adj_t

        # Next frontier: newly reached nodes (clamped to [0, 1] as probabilities).
        next_frontier = torch.clamp(scores, 0.0, 1.0)

        # Update visited with maximum (logical OR in probability space).
        visited = torch.maximum(visited, next_frontier)

        # Move frontier forward for the next iteration.
        frontier = next_frontier

    return visited.detach().cpu().numpy()


def main():
    task = load_task(__file__)

    # The evaluator handles the canonical batch, hop sweep, and baseline.
    payload = task.evaluate(model_fn)

    # Write benchmark.json alongside any other artefacts.
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark payload written to: {run_dir}")


if __name__ == "__main__":
    main()