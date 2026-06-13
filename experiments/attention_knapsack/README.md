# attention_knapsack

## Question

Can a model (or an attention-based mechanism) learn to **select a subset of
items that maximises total value under a shared capacity constraint** — i.e.
solve the classic 0/1 knapsack problem — and how robustly does that ability
hold as the capacity budget tightens or loosens?

A knapsack solution requires the mechanism to weigh each item's value against
its weight *and* against a global resource budget shared across all items —
a non-local, combinatorial decision. This goal measures how close an attempt
gets to the exact optimum, how often it respects the capacity constraint, and
how its quality degrades across the capacity sweep.

## Setup

**Synthetic generator.** Each instance has `n_items = 16` items with integer
weights and values drawn i.i.d. from `[1, 10]`. The shared `capacity` is a
fixed fraction of the expected total weight
(`capacity = capacity_frac · n_items · (w_max+1)/2`). Ground-truth optimal
selections and optimal values are computed by an exact integer DP solver
(`task._solve_knapsack_exact`).

`generate(seed)` is **deterministic**: same seed → same batch. `evaluate`
always uses `seed=42` for both the canonical batch and every sweep batch, so
two attempts at this goal see identical data.

## Canonical measurement condition

| parameter       | value |
|-----------------|-------|
| `batch_size`    | 256   |
| `n_items`       | 16    |
| `w_max`         | 10    |
| `v_max`         | 10    |
| `capacity_frac` | **0.5** (canonical) |
| seed            | 42    |

The capacity sweep evaluates `capacity_frac ∈ {0.3, 0.4, 0.5, 0.6, 0.7}`.

## The `model_fn` contract

Attempts hand `evaluate` a single callable:

```python
model_fn(weights, values, capacity) -> selection
```

| arg / return | type | semantics |
|--------------|------|-----------|
| `weights`    | `np.ndarray (batch_size, n_items)` float32 | per-item weights |
| `values`     | `np.ndarray (batch_size, n_items)` float32 | per-item values |
| `capacity`   | `float` | shared capacity for the whole batch |
| **returns**  | `np.ndarray (batch_size, n_items)` in `[0, 1]` | per-item selection score; `evaluate` thresholds at `0.5` |

Attempts never build the payload themselves — they call
`task.evaluate(model_fn)` and record what it returns.

`task.random_model_fn()` is a **factory**: it returns a `model_fn` emitting
uniform random selection scores (pure NumPy). The smoke test runs
`benchmark.score(task.evaluate(task.random_model_fn()))`.

## Payload contract

`task.evaluate` returns:

```python
{
  "version": 1,
  "config": {batch_size, n_items, w_max, v_max, capacity_frac, sweep_fracs},
  "canonical":          <record>,            # at capacity_frac = 0.5
  "sweep":          [<record> × 5],          # one per sweep frac, ordered
  "baseline_canonical": <record>,            # greedy baseline, same batch
  "baseline_sweep": [<record> × 5],          # greedy baseline, per frac
}
```

Each `<record>` is a dict of pre-aggregated scalars (means over the batch):

| key              | type  | meaning |
|------------------|-------|---------|
| `capacity_frac`  | float | the fraction used for this batch |
| `optimal_value`  | float | mean exact-optimal value |
| `model_value`    | float | mean attained value — **0 for infeasible instances** |
| `model_weight`   | float | mean selected weight |
| `feasible_rate`  | float | fraction of instances respecting capacity |
| `optimality_gap` | float | `1 − model_value / optimal_value` |

Feasibility is folded into value: an over-capacity selection contributes 0
value, so `optimality_gap` penalises constraint violations directly.

## Metrics

`benchmark.py` (`VERSION = 1`) returns a flat dict. Optimality is
`1 − gap`, clamped to `[0, 1]`. **All metrics are bigger-is-better.**

| metric | class | meaning |
|--------|-------|---------|
| `knapsack_optimality_robustness` | **headline** | mean optimality across the 5-frac sweep |
| `knapsack_optimality_canonical`  | canonical | optimality at `capacity_frac = 0.5` |
| `knapsack_feasible_canonical`    | canonical | feasible rate at canonical |
| `knapsack_optimality_cap_0p3 … 0p7` | per-slice | optimality at each sweep frac |
| `knapsack_feasible_cap_0p3 … 0p7`   | per-slice | feasible rate at each sweep frac |
| `linear_baseline_optimality_canonical` | baseline | greedy ratio heuristic at canonical |
| `linear_baseline_optimality_cap_0p3 … 0p7` | baseline | greedy heuristic per slice |
| `lift_over_linear_baseline`      | contrast | canonical optimality − greedy baseline |

The **greedy value/weight-ratio heuristic** is the no-mechanism reference,
measured on the identical batches. Beating it is what makes an attempt
meaningful.

### Pipeline hooks

- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken` fires (skipping the jury) on NaN/inf, when canonical
  optimality fails to beat the greedy baseline, or when feasibility is near
  zero — never on a borderline-but-real result.

## Bump procedure

Bump `VERSION` (and update this contract in the same commit) when you change
any metric formula, rename/retype a payload key, or change the canonical
condition (e.g. `capacity_frac`, `n_items`, the sweep fracs). Adding a new
metric or an optional payload key does not require a bump.
