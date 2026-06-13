# attention_astar

## Question
Does attention implement A*-like heuristic search when solving grid navigation
tasks? Specifically, does the attention distribution over grid positions place
its mass on cells with **low A\* f-value** (`f = g + h`, where `g` is distance
from the agent and `h` is the heuristic distance to the goal), and does its peak
fall on the cell A\* would actually expand next?

## Setup
**Synthetic generator only.** No trained model required.

- **Grid**: Square grid of size `N x N` (canonical `N=8`).
- **Objects**: One start cell (S), one goal cell (G), and obstacle cells (X)
  placed independently at random with probability `density`. Start, goal, and
  the agent's current cell are always free, distinct, and connected (grids with
  no agent→goal path are resampled out).
- **Task**: The model receives a 4-channel grid and outputs an attention
  distribution over all `N*N` positions, interpreted as "where the model
  attends for the next step" from the agent's current cell. We measure whether
  that distribution aligns with A\* search.

**Canonical measurement condition**:
- Grid size: `8x8`
- Canonical obstacle density: `0.2` (index 2 in the sweep)
- Density sweep: `[0.0, 0.1, 0.2, 0.3, 0.4]`
- Heuristic: Manhattan distance
- Number of test grids per evaluation: 125 (25 per density across the 5 densities)
- Seed: 42 (fixed inside `task.evaluate`; `generate(seed)` is deterministic per seed)

## Model function signature
```python
ModelFn = Callable[[np.ndarray], np.ndarray]

def model_fn(grids: np.ndarray) -> np.ndarray:
    """
    Args:
        grids: float32 array of shape [B, N, N, C] where C=4 channels:
            channel 0: obstacle mask (1.0 = obstacle, 0.0 = free)
            channel 1: start position (one-hot)
            channel 2: goal position (one-hot)
            channel 3: agent / current position (one-hot)
    Returns:
        attention: float32 array of shape [B, N, N]. Non-negative finite
        weights over grid cells for each batch item. `task.evaluate`
        re-normalises each item to sum to 1 over the (N, N) plane, so the model
        need not normalise itself.
    """
    ...
```
The attempt's `main.py` must implement this callable. It may use any internal
architecture (Transformer, MLP, etc.) but must accept and return NumPy arrays
with the exact shapes above.

`task.random_model_fn()` returns a reference `ModelFn` that emits **uniform**
attention over free cells — the no-mechanism baseline used by the smoke test.

## Payload contract
`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                 # int, must equal benchmark.VERSION
    "grid_size": 8,               # int, canonical N
    "heuristic": "manhattan",     # str
    "num_grids": 125,             # int, total grids actually evaluated
    "canonical_density_index": 2, # int, index into `sweep` for density 0.2
    "sweep": [                    # list[dict], one record per density, ascending
        {
            "obstacle_density": 0.0,          # float, the density for this slice
            "n_grids": 25,                    # int, grids in this slice (> 0)
            "attention_entropy": 0.123,       # float, mean attention entropy (nats)
            "heuristic_alignment": 0.456,     # float in [-1, 1], mean Spearman ρ
                                              #   between attention mass and -f_value
            "top1_optimal_rate": 0.789,       # float in [0, 1]
            "top3_optimal_rate": 0.890,       # float in [0, 1]
            "path_optimality_gap": 0.111,     # float >= 0, mean relative path excess
            "linear_baseline_alignment": 0.0, # float in [-1, 1], uniform-attention
                                              #   alignment on the SAME grids
        },
        ...  # densities 0.1, 0.2, 0.3, 0.4
    ],
}
```

All floats are Python `float` (never numpy scalars); all ints are Python `int`.

Semantics:
- `attention_entropy` — entropy in **nats** of the normalised attention plane,
  averaged over the slice. Lower = more focused.
- `heuristic_alignment` — Spearman rank correlation between the attention mass
  at each free cell and that cell's **negative** A\* f-value, averaged over the
  slice. Higher = more mass on lower-f (better) cells. Degenerate (constant)
  attention yields 0.
- `top1_optimal_rate` / `top3_optimal_rate` — fraction of grids where the
  argmax / top-3 attention cells include an **optimal next neighbour** of the
  agent (a neighbour lying on at least one optimal A\* path).
- `path_optimality_gap` — greedily follow top-1 attention (masking obstacles
  and visited cells) up to a step cap; gap is `(model_steps - astar_steps) /
  astar_steps`, averaged. Failure to reach the goal scores a gap of `1.0`.
- `linear_baseline_alignment` — the same `heuristic_alignment` computation run
  on **uniform** attention over the identical grids; the no-mechanism reference.

## Metrics
`benchmark.score(payload)` returns a flat dict:

| Metric | Source | Direction | Meaning |
|--------|--------|-----------|---------|
| `version` | `payload["version"]` | — | Protocol version |
| `astar_alignment_canonical` | `sweep[canonical].heuristic_alignment` | **Bigger** | **Headline**: alignment at density 0.2 |
| `astar_alignment_density_0p0` | `sweep[0].heuristic_alignment` | Bigger | Alignment, no obstacles |
| `astar_alignment_density_0p1` | `sweep[1].heuristic_alignment` | Bigger | |
| `astar_alignment_density_0p2` | `sweep[2].heuristic_alignment` | Bigger | |
| `astar_alignment_density_0p3` | `sweep[3].heuristic_alignment` | Bigger | |
| `astar_alignment_density_0p4` | `sweep[4].heuristic_alignment` | Bigger | |
| `astar_entropy_density_0p2` | `sweep[2].attention_entropy` | **Smaller** | Focus at canonical |
| `top1_optimal_canonical` | `sweep[2].top1_optimal_rate` | Bigger | Top-1 hits optimal neighbour |
| `top3_optimal_canonical` | `sweep[2].top3_optimal_rate` | Bigger | Top-3 contains optimal neighbour |
| `path_gap_canonical` | `sweep[2].path_optimality_gap` | Smaller | Relative path-length excess |
| `density_robustness` | `min(alignment)/max(alignment)` across sweep, clipped `[0,1]` | Bigger | Alignment retained at hardest density |
| `linear_baseline_alignment_density_0p2` | `sweep[2].linear_baseline_alignment` | Bigger | Uniform-attention reference |
| `lift_over_baseline_canonical` | `astar_alignment_canonical − linear_baseline_alignment_density_0p2` | Bigger | Improvement over uniform |

Per-slice metrics use the `0pX` naming convention (`0.2` → `0p2`).

## Bump procedure
Increment `VERSION` in `benchmark.py` if:
- any metric formula changes;
- a payload key is added, removed, or retyped;
- the canonical grid size, heuristic, density, or sweep changes.

After bumping, update this README's "Payload contract" and "Metrics" tables in
the same commit. Old `benchmark.json` files stay on disk; the dashboard filters
to the highest `version`.

## Smoke test
The pipeline runs, before any attempt:
```python
payload  = task.evaluate(task.random_model_fn())
metrics  = benchmark.score(payload)
```
`task.random_model_fn()` returns a `ModelFn` emitting uniform attention. All
three calls must complete without error or the goal is rejected.
