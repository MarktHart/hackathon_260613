# attention_topo_sort

## Question

Can an attention mechanism encode the **partial order** of a DAG so that a
**topological sort** falls out of its attention pattern?

A topological sort of a DAG requires that every node appear *after* all of its
ancestors. If a single attention layer is doing the work of a topo-sort, then
for any pair of nodes related by the partial order, the *later* node (the
descendant) should attend **back** to the *earlier* node (the ancestor) more
than the reverse. This goal measures how faithfully an attempt's attention
matrix respects the ancestor→descendant direction of a DAG.

## Setup

**Fully synthetic.** No trained model, no external dataset. `task.generate`
samples random DAGs (acyclic by construction) at several edge densities. Each
attempt supplies a `model_fn` that, given a DAG's adjacency matrix, produces a
square attention matrix over its nodes. The task evaluator measures, per DAG,
the fraction of ancestor/descendant pairs whose attention respects the partial
order.

Because the data is fully determined by the seed and the canonical config,
`generate(seed)` is deterministic; two attempts at this goal see the exact same
DAGs.

## Canonical measurement condition

| knob              | value                         |
|-------------------|-------------------------------|
| number of nodes   | `N_NODES = 8`                 |
| DAGs per density  | `N_DAGS = 24`                 |
| density sweep     | `[0.1, 0.2, 0.3, 0.5]`        |
| canonical density | `CANONICAL_DENSITY = 0.3`     |
| eval seed         | `EVAL_SEED = 0`               |

The **canonical** headline is measured at density `0.3`.

## The `model_fn` contract

```python
ModelFn = Callable[[np.ndarray, int], np.ndarray]

def model_fn(adjacency: np.ndarray, n: int) -> np.ndarray:
    """
    adjacency : (n, n) float/bool array. adjacency[i, j] == 1 means a directed
                edge i -> j, i.e. i must come before j in any topological order.
                The graph is acyclic and has a zero diagonal.
    n         : number of nodes (== adjacency.shape[0]), passed for convenience.

    returns   : (n, n) array of attention weights. Row r is the attention
                distribution of query node r over all key nodes. Weights should
                be non-negative; rows are renormalised defensively by the
                evaluator, so they need not sum to 1 exactly. Negative entries
                are clipped to 0.
    """
```

The evaluator never inspects anything but this returned matrix. An attempt
that wants to "do a topo sort" should make descendant rows place more mass on
ancestor columns than ancestor rows place on descendant columns.

## Payload contract

`task.evaluate(model_fn)` returns:

```python
{
    "canonical_density": float,        # == CANONICAL_DENSITY
    "n_nodes": int,                    # == N_NODES
    "n_dags": int,                     # DAGs per density slice
    "model_name": str,                 # free-form label, self-describing
    "sweep": [                         # one record per density, in sweep order
        {
            "density": float,          # nominal edge density of this slice
            "topo_respect": float,     # in [0, 1]; fraction of ordered pairs
                                       #   (ancestor a, descendant d) for which
                                       #   attn[d, a] > attn[a, d] (ties = 0.5)
            "uniform_respect": float,  # same metric for a uniform attention
                                       #   matrix under identical DAGs (== 0.5)
            "pairs": int,              # number of ordered ancestor pairs scored
        },
        ...
    ],
}
```

`benchmark.score` consumes exactly this shape. The `version` key is **not** in
the payload — `benchmark.score` stamps it onto the metrics.

## Metrics

All metrics are flat scalars. Bigger is better for every `topo_respect` /
`lift` / `robustness` metric.

| metric                              | meaning                                                            |
|-------------------------------------|--------------------------------------------------------------------|
| `version`                           | benchmark version (first key)                                      |
| `topo_respect_canonical`            | headline: `topo_respect` at the canonical density                  |
| `topo_respect_density_0pX`          | per-slice topo respect at density `0.X`                            |
| `uniform_baseline_density_0pX`      | uniform-attention reference at density `0.X` (≈ 0.5)              |
| `uniform_baseline_canonical`        | uniform-attention reference at the canonical density (≈ 0.5)      |
| `lift_over_uniform_density_0pX`     | `topo_respect − uniform_respect` at density `0.X`                 |
| `lift_over_uniform_canonical`       | lift at the canonical density                                      |
| `topo_robustness`                   | worst-slice normalised skill, in `[0, 1]` (see below)             |

**`topo_respect`** ∈ `[0, 1]`. A perfect topo-sort attention scores `1.0`;
uniform / random scores `0.5`. Per ordered pair `(a, d)` with `a` an ancestor
of `d`: credit `1` if `attn[d, a] > attn[a, d]`, `0.5` on a tie, `0` otherwise.

**`topo_robustness`** ∈ `[0, 1]` — the headline summary. The *worst* per-slice
normalised skill, where the normalised skill of a slice is
`clip((topo_respect − 0.5) / 0.5, 0, 1)`. A method that beats chance everywhere
scores high; one that collapses on the hardest density scores near 0. This is
the single number to optimise.

## Bump procedure

Bump `VERSION` (currently `1`) when you change any existing metric formula,
rename/retype a payload key, or change the canonical density / node count /
sweep semantics. Adding a new density slice or a new derived metric does **not**
require a bump (the sweep is already extensible and per-slice keys are derived
from the data). After bumping, update the payload contract and metrics tables
above in the same commit. Old `benchmark.json` files stay on disk; the
dashboard filters to the highest version present.
