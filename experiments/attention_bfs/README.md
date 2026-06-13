# attention_bfs: Can a mechanism propagate reachability like BFS?

## Question

A single application of attention can only move information **one hop** across a
graph. Breadth-first search reaches distance-`h` nodes only after `h` rounds of
propagation. This goal asks: does an attempt's mechanism implement genuine
**multi-hop** reachability propagation, or does it collapse to a one-shot,
single-hop lookup?

Concretely: given a directed graph, a source node, and a hop budget `h`, predict
which nodes are reachable from the source in **at most `h` steps**. A real
BFS-like mechanism keeps its accuracy high as `h` grows; a one-hop strawman does
not.

## Setup

**Synthetic generator only.** No trained checkpoint or external dataset is
required. The goal evaluates whatever `model_fn` an attempt provides — a trained
transformer read-out, a hand-coded attention circuit, or a baseline. The
attempt's job is to turn its mechanism into the narrow `model_fn` callback below.

- **Graphs**: directed Erdős–Rényi `G(N, p)` with `N = 24`, `p = 0.10`, no self
  loops. Sparse enough that BFS distances span the full hop axis.
- **Batch**: `48` graphs, each with one randomly chosen source node.
- **Determinism**: `generate(seed)` is fully deterministic. The canonical batch
  uses `seed = 0`.

## `model_fn` contract

```python
model_fn(adjacency: np.ndarray, source: int, hops: int) -> np.ndarray
```

| arg | type | meaning |
|-----|------|---------|
| `adjacency` | `(N, N)` array, 0/1 | `adjacency[i, j] == 1` iff directed edge `i -> j`; no self loops |
| `source` | `int` in `[0, N)` | BFS source node |
| `hops` | `int >= 1` | hop budget `h` |
| **returns** | `(N,)` float array | per-node reachability **probabilities** in `[0, 1]`; entry `k` = P(node `k` reachable from `source` in `<= hops` steps) |

The evaluator thresholds the returned probabilities at `0.5`. Attempts never
build the payload themselves — they hand `model_fn` to `task.evaluate`.

## Canonical measurement condition

- `N_NODES = 24`, `EDGE_PROB = 0.10`, `N_GRAPHS = 48`, `seed = 0`.
- Hop sweep axis: `hops ∈ {1, 2, 3, 4, 5}`.
- **Canonical hops = 5** (the full BFS horizon); the headline metric is measured
  here.
- Decision threshold `0.5`. Metrics are pooled (micro-averaged) over all nodes
  of all graphs at a given hop budget.

## Payload contract

`task.evaluate(model_fn)` returns exactly:

```python
{
    "version": 1,                 # payload schema version (== benchmark.VERSION)
    "n_graphs": 48,
    "n_nodes": 24,
    "edge_prob": 0.10,
    "canonical_hops": 5,
    "hops_axis": [1, 2, 3, 4, 5],
    "threshold": 0.5,
    "sweep": [                    # one record per hop budget, in axis order
        {
            "hops": 1,
            "model_f1": 0.0,        # pooled F1 of model prediction vs BFS truth
            "model_acc": 0.0,       # pooled node-level accuracy
            "model_precision": 0.0,
            "model_recall": 0.0,
            "baseline_f1": 0.0,     # pooled F1 of the 1-hop strawman
            "baseline_acc": 0.0,
        },
        # ... one record for each hops value ...
    ],
}
```

All values are pre-aggregated scalars — no tensors. The **baseline** (`source +
direct neighbours`, regardless of `h`) is computed inside `evaluate` under
identical conditions so the contrast is apples-to-apples.

## Metrics

`benchmark.score(payload)` returns a flat dict (`version` first):

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `version` | `VERSION` | — | dashboard version filter |
| `bfs_f1_canonical` | pooled F1 at `hops = 5` | **bigger** | **headline** — reachability quality at the full BFS horizon |
| `bfs_acc_canonical` | pooled accuracy at `hops = 5` | bigger | node-level accuracy at the horizon |
| `bfs_f1_mean` | mean `model_f1` over the sweep | bigger | overall quality across hop budgets |
| `bfs_reachability_robustness` | `f1(hops=5) / f1(hops=1)`, clamped `[0,1]` | bigger | how well F1 holds as propagation depth grows |
| `bfs_f1_hops_<h>` | pooled F1 at hop budget `h` | bigger | per-slice F1 |
| `bfs_acc_hops_<h>` | pooled accuracy at hop budget `h` | bigger | per-slice accuracy |
| `linear_baseline_f1_hops_<h>` | 1-hop strawman F1 at `h` | bigger | per-slice baseline |
| `linear_baseline_f1_canonical` | 1-hop strawman F1 at `hops = 5` | bigger | baseline at the horizon |
| `lift_over_linear_baseline` | `bfs_f1_canonical − linear_baseline_f1_canonical` | **bigger** | improvement over the no-mechanism reference |

All metrics share a consistent direction: **bigger is better.** A meaningful
attempt must keep F1 high as `hops` grows (high `bfs_reachability_robustness`)
and beat the 1-hop baseline at the horizon (positive `lift_over_linear_baseline`).

### Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run a real model on the GPU; `task`/`benchmark`
  stay pure CPU/NumPy.
- `is_obviously_broken(metrics)` returns `True` (skips the jury) on NaN/inf, on
  `bfs_f1_canonical` outside `[0, 1]`, or when the method fails to beat the 1-hop
  baseline at the canonical horizon.

## Bump procedure

Bump `VERSION` in `benchmark.py` (and `version` in `task.py`) when you change a
metric formula, rename/retype a payload key, or change the canonical condition
(`N`, `p`, hop axis, canonical hops, threshold). Update this README's payload and
metric tables in the same commit. Old `benchmark.json` files stay on disk; the
dashboard filters to the highest `VERSION`.
