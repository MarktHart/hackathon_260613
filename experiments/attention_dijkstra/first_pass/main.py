


import torch
import numpy as np

DEVICE = "cuda"

def model_fn(weights: np.ndarray, source: int) -> np.ndarray:
    """Predict shortest-path distances using iterative relaxation with attention.

    weights: (n, n) adjacency matrix of an undirected, positively-weighted graph.
    source: index of the single source node.
    returns: (n,) array of predicted shortest distances.
    """
    n = weights.shape[0]
    # Weights: adjacency matrix, inf for no edge.
    weights_t = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)

    # Replace inf with a large number (but not numerically infinite).
    weights_t = torch.where(weights_t == float('inf'), torch.tensor(1e9, device=DEVICE), weights_t)

    # Initialize distances.
    dist = torch.full((n,), float('inf'), dtype=torch.float32, device=DEVICE)
    dist[source] = 0.0

    # Iterative attention (soft-min relaxation over 10 hops).
    num_iterations = 10

    for _ in range(num_iterations):
        # dist.unsqueeze(1) => (n, 1)
        # weights_t => (n, n)
        # dist_up => (n, n)
        # where dist_up[u,v] = current best distance to u + edge cost from u to v
        dist_up = dist.unsqueeze(1) + weights_t

        # For each node v, the new best distance is min over u of (dist_up[u,v]).
        # This is the forward pass of one Bellman-Ford / attention hop.
        dist, _ = torch.min(dist_up, dim=0)  # dist now shape (n,)

    # Return distances as numpy array.
    return dist.cpu().numpy()



from agentic.experiments import load_task, record_benchmark, results_dir

def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir(__file__), payload)


if __name__ == "__main__":
    main()


