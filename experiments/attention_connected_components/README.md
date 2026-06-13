# attention_connected_components

## Question

Does an attention mechanism recover the **connected components** of a graph —
the *transitive closure* of the edge relation — or does it only ever express
the **1-hop adjacency** relation it can read off in a single attention step?

Two nodes are in the same connected component if a path of *any* length links
them. A single attention application propagates information one hop. So the
interesting mechanistic claim is whether an attempt's attention matrix marks
nodes as "same component" even when they are many hops apart in the graph. The
adjacency-only baseline cannot: it sees the direct edge or nothing.

## Setup

**Synthetic.** Each graph is a disjoint union of `K = 4` *path* components.
A path of `diameter + 1` nodes has graph diameter `diameter`, which gives a
clean knob: at `diameter = 1` each component is a single edge (adjacency ==
closure), and as `diameter` grows the gap between the 1-hop relation and the
true closure widens. Node indices are randomly permuted so adjacency is not
trivially block-structured.

The sweep axis is component **diameter** ∈ `{1, 2, 3, 5}`, with the canonical
condition at **diameter 3**. Each slice averages over `NUM_GRAPHS = 8` graphs.

## Canonical measurement condition

`task.generate(seed=0)` is the canonical fixture and `task.evaluate` always
uses it. Generation is deterministic: same seed → identical graphs. Do not
re-roll seeds; compare attempts on the fixed batch.

## Model contract (`model_fn`)

```python
model_fn(adjacency: np.ndarray) -> np.ndarray
```

| arg / return | shape    | type  | semantics                                                          |
|--------------|----------|-------|--------------------------------------------------------------------|
| `adjacency`  | `(N, N)` | float | symmetric, 0/1, zero diagonal; the undirected graph                |
| returns      | `(N, N)` | float | same-component **affinity**; `affinity[i,j]` = belief i,j share a component |

The evaluator thresholds the returned affinity at `0.5` to obtain a boolean
"same component?" prediction for every unordered pair `i < j`. Only the
`>= 0.5` relation is read; the diagonal is ignored. Affinities may be any
reals. `N` varies per slice (`N = K * (diameter + 1)`); the function must
handle whatever `N` it is handed.

## Payload contract

`task.evaluate(model_fn)` returns exactly:

```python
{
  "version": 1,
  "canonical_diameter": 3,
  "num_components": 4,
  "num_graphs": 8,
  "sweep": [
    {
      "diameter": <int>,
      "model":    {"tp": int, "fp": int, "fn": int, "tn": int},
      "baseline": {"tp": int, "fp": int, "fn": int, "tn": int},
    },
    ...   # one record per diameter in {1, 2, 3, 5}
  ],
}
```

`model` counts come from the attempt's affinity; `baseline` counts come from
the raw adjacency matrix (the 1-hop reference) under identical conditions.
Counts are over unordered node pairs (`i < j`) aggregated across the slice's
graphs. The payload carries confusion counts, not F1 — `benchmark.score`
computes every rate so the formula lives in one place.

## Metrics

`benchmark.score(payload)` (pure Python) returns a flat scalar dict. Pairwise
F1 of the same-component relation is `2·tp / (2·tp + fp + fn)`, defined as
`0.0` when the denominator is zero. All F1 values are bigger-is-better in
`[0, 1]`.

| metric                              | meaning                                                        |
|-------------------------------------|----------------------------------------------------------------|
| `version`                           | always first; dashboard filters to the highest version present |
| `transitive_closure_robustness`     | **headline**: mean model F1 across the diameter sweep, `[0,1]` |
| `cc_f1_canonical`                   | model F1 at diameter 3                                          |
| `cc_f1_diam_<D>`                    | per-slice model F1 at diameter `D`                             |
| `adjacency_baseline_f1_diam_<D>`    | per-slice 1-hop baseline F1 at diameter `D`                    |
| `adjacency_baseline_f1_canonical`   | baseline F1 at diameter 3                                      |
| `lift_over_adjacency_diam_<D>`      | model F1 − baseline F1 at diameter `D`                         |
| `lift_over_adjacency_canonical`     | model − baseline at diameter 3                                 |
| `mean_lift_over_adjacency`          | mean of `lift_over_adjacency_diam_*` across the sweep          |

How to read it: the adjacency baseline is ~1.0 at `diameter = 1` and falls as
diameter grows (it misses every non-adjacent same-component pair). A model that
truly computes the closure holds its F1 flat across the sweep, so
`transitive_closure_robustness` stays near 1 and `mean_lift_over_adjacency`
grows positive. A model that merely copies adjacency tracks the baseline and
shows ~0 lift.

### Pipeline hooks

- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken(metrics)` → `True` on any NaN/inf, or when
  `cc_f1_canonical <= adjacency_baseline_f1_canonical` (failing to beat the
  no-mechanism baseline at the canonical condition). It only short-circuits the
  jury; it never affirms a result.

## Bump procedure

`VERSION = 1`. Bump `benchmark.VERSION` (and `task`'s payload `version`) when
you change any existing metric formula, rename/retype/remove a payload key, or
move the canonical diameter. You need **not** bump to add a new metric, add an
optional payload key with a default, or extend the diameter sweep. On a bump,
update this contract in the same commit; old `benchmark.json` files stay on
disk and the dashboard hides them behind the version filter.
