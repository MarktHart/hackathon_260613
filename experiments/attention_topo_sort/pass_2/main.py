import torch
import numpy as np

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

def model_fn(adjacency: np.ndarray, n: int) -> np.ndarray:
    """
    Hand-built attention circuit: compute transitive closure of the DAG on GPU
    and return attention where each node attends to its ancestors.

    Args:
        adjacency: (n, n) bool/float array, adjacency[i, j] == 1 means edge i -> j
                   (i precedes j, i is ancestor of j)
        n: number of nodes

    Returns:
        (n, n) attention matrix. Row d (descendant) has mass on column a (ancestor)
        iff there is a directed path a -> ... -> d.
    """
    # Move to GPU
    adj = torch.as_tensor(adjacency, dtype=torch.bool, device=DEVICE)

    # Transitive closure via Floyd-Warshall on GPU (n=8 is tiny, so this is fast)
    # reach[a, d] = True iff a is an ancestor of d
    reach = adj.clone()
    for k in range(n):
        # reach[a, d] |= reach[a, k] & reach[k, d]
        reach = reach | (reach[:, k:k+1] & reach[k:k+1, :])

    # Attention: descendant d attends to ancestor a
    # So attn[d, a] = reach[a, d] (transpose of reachability)
    attn = reach.T.to(torch.float32)

    # Add small self-attention and background so rows aren't degenerate
    # (evaluator renormalizes anyway, but this gives non-zero everywhere)
    attn = attn + 0.1 * torch.ones((n, n), device=DEVICE)
    attn[torch.arange(n, device=DEVICE), torch.arange(n, device=DEVICE)] += 1.0

    return attn.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Also save the attention matrices for the canonical density for visualization
    batch = task.generate(task.EVAL_SEED)  # access canonical batch
    canonical_idx = list(batch.densities).index(batch.canonical_density)
    canonical_dags = batch.dags[canonical_idx]

    all_attn = []
    for adj in canonical_dags:
        all_attn.append(model_fn(adj, adj.shape[0]))
    all_attn = np.stack(all_attn)  # (n_dags, n, n)

    np.save(run_dir / "canonical_attention.npy", all_attn)
    np.save(run_dir / "canonical_adjacency.npy", np.stack(canonical_dags))

    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()