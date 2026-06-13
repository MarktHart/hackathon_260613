# attention_dijkstra

## Question

Can an attention-style mechanism compute **single-source shortest-path
distances** on a weighted graph — i.e. does an iterated soft-min relaxation
(the way Dijkstra/Bellman–Ford propagate distances through neighbours)
recover the true shortest-path distances from a source node to every other
node? We measure how accurately an attempt's predicted distances match the
exact Dijkstra distances, and how well that accuracy holds as graphs grow
(more relaxation hops required).

## Setup

**Synthetic generator only.** No trained model is involved. Each example is a
connected, undirected, positively-weighted graph with a single designated
source node. Ground-truth distances are computed exactly with Dijkstra's
algorithm.

- Graph size: `n_nodes = 16` (canonical), swept over `{8, 16, 32, 64}`.
- Graphs are built from a random spanning tree (guarantees connectivity) plus
  `~n` extra random edges (creates alternative multi-hop paths).
- Edge weights: uniform in `[1, 10]` (continuous).
- Source: one node chosen uniformly at random per graph.
- `N_SEEDS = 20` independent graphs per sweep slice; metrics are averaged over
  seeds.

## Canonical measurement condition

```
n_nodes      = 16
n_seeds      = 20
weight_range = (1.0, 10.0)
rel_tol      = 0.10        # accuracy tolerance (see below)
eval_seed    = 42
```

Every attempt evaluates on this exact batch (`task.generate(seed=42)` over the
full `n_nodes` sweep). The canonical slice is `n_nodes = 16`.

## Model function signature

This is the goal's contract with attempts. Keep it narrow:

```python
def model_fn(weights: np.ndarray, source: int) -> np.ndarray:
    """
    weights : (n, n) float64 symmetric adjacency. weights[i, j] is the cost of
              edge (i, j), np.inf where no edge exists, 0.0 on the diagonal.
    source  : index of the single source node.
    returns : (n,) float array of predicted shortest-path distances from
              `source` to every node. predicted[source] should be ~0.
    """
```

- The attempt's mechanism (iterated attention relaxation, a learned head,
  etc.) is whatever produces those distances. `task.py` only sees the numbers.
- Predictions on unreachable nodes are not scored (all generated graphs are
  connected, so every non-source node is reachable). Non-finite predictions on
  scored nodes are treated as wrong, not as errors.

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                       # payload schema version (== benchmark.VERSION)
    "model_name": str,                  # label, echoed for debugging
    "canonical_n": 16,                  # canonical slice
    "n_nodes_sweep": [8, 16, 32, 64],   # sweep axis values, in order
    "rel_tol": 0.10,                    # relative tolerance used for accuracy
    "sweep": [                          # one record per n_nodes, same order as n_nodes_sweep
        {
            "n_nodes": int,
            "distance_accuracy": float,   # mean over seeds: fraction of reachable
                                          #   non-source nodes within tolerance
            "order_correlation": float,   # mean Spearman corr (pred vs true), in [-1, 1]
            "n_seeds": int,
        },
        ...
    ],
    "linear_baseline": [                # one-hop strawman, SAME graphs/seeds
        {
            "n_nodes": int,
            "distance_accuracy": float,
            "order_correlation": float,
            "n_seeds": int,
        },
        ...
    ],
}
```

**Definitions (per graph, then averaged over seeds in the slice):**

- Let `true[v]` be the exact Dijkstra distance from `source` to `v`. The scored
  set is the reachable, non-source nodes.
- **distance_accuracy**: fraction of scored nodes `v` with
  `|pred[v] - true[v]| <= rel_tol * |true[v]| + 1e-6`.
- **order_correlation**: Spearman rank correlation between `pred` and `true`
  over the scored nodes (0 if either side is constant). A scale-invariant check
  that the mechanism at least orders nodes by distance.
- **linear_baseline (one-hop)**: `pred[v] =` the direct edge weight `source→v`
  (`inf` if no direct edge), `pred[source] = 0`. This is shortest paths
  truncated to a single relaxation step — correct on direct neighbours, wrong
  everywhere multi-hop. It is the no-propagation reference.

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars (`version` first):

| Key | Formula | Direction | Notes |
|-----|---------|-----------|-------|
| `version` | `payload["version"]` | — | Always first key. |
| `dijkstra_robustness` | `distance_accuracy(n=64) / distance_accuracy(n=8)`, clamped to `[0, 1]` | bigger ↑ | **Headline.** How well accuracy survives graph growth. |
| `distance_accuracy_canonical` | `distance_accuracy` at `n_nodes=16` | bigger ↑ | Headline accuracy. |
| `order_correlation_canonical` | `order_correlation` at `n_nodes=16` | bigger ↑ | Ordering quality. |
| `lift_over_baseline_canonical` | `distance_accuracy_canonical − onehop_baseline_accuracy_n_16` | bigger ↑ | Beats the no-propagation strawman? |
| `distance_accuracy_n_<n>` | `distance_accuracy` at that `n_nodes` | bigger ↑ | Per-slice (`n` ∈ {8,16,32,64}). |
| `order_correlation_n_<n>` | `order_correlation` at that `n_nodes` | bigger ↑ | Per-slice. |
| `onehop_baseline_accuracy_n_<n>` | baseline `distance_accuracy` at that `n_nodes` | bigger ↑ | Reference, same graphs. |

Direction of better is consistent: every metric here is bigger-is-better.

**Edge cases.** Empty sweeps / length mismatches raise `ValueError`. Zero
denominators (e.g. `distance_accuracy(n=8) == 0`) make `dijkstra_robustness`
fall back to `0.0`. Spearman returns `0.0` when a side is constant. `score()`
clamps the headline to `[0, 1]`.

**Pipeline hooks.** `GPU_REQUIREMENT = 1`. `is_obviously_broken` returns `True`
on any NaN/inf metric, or when `distance_accuracy_canonical` fails to exceed the
one-hop baseline at the canonical slice (no path-finding learned → skip jury).

## Bump procedure

Bump `VERSION` in `benchmark.py` and `version` in the payload **together** when:

- any existing metric formula changes;
- a payload key is renamed, removed, or retyped;
- the canonical condition (`canonical_n`, `rel_tol`, weight range, seed) changes.

You do **not** need to bump when adding a new metric without touching existing
ones, adding an optional payload key with a default, or adding a new `n_nodes`
slice to the sweep. After bumping, update this README's Payload contract and
Metrics table in the same commit.
