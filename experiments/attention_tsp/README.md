# Attention TSP: Nearest-Neighbour Routing Heuristic

## Question
Can an attention mechanism implement the nearest-neighbour (NN) TSP routing
heuristic? At each decode step the mechanism sees the full city layout, the
current city, and which cities have already been visited, and must point its
attention at the **nearest unvisited city**. We measure how faithfully a given
mechanism reproduces this greedy heuristic, and how well that fidelity holds as
the number of cities grows.

## Setup
- **Data (canonical):** Fully synthetic random Euclidean TSP instances — city
  coordinates drawn uniformly from the unit square `[0, 1]^2`. No downloaded
  checkpoint or dataset is required; everything is generated from a seed.
- **Sweep axis:** problem size `N ∈ {5, 10, 20, 40}` cities
  (`task.N_CITIES_SWEEP`).
- **Instances:** `N_INSTANCES = 20` random instances per problem size.
- **Decoding:** Tours always start at city `0` (`START_CITY`). At each step the
  evaluator calls `model_fn`, masks out already-visited cities, and takes the
  `argmax` of the returned logits as the next city.
- **Baseline:** A random city-selection policy run under identical conditions
  (uniform choice among unvisited cities), giving both an expected step-wise
  accuracy and an achieved tour length.

`task.generate(seed)` is deterministic — the same seed yields identical
coordinates. `task.evaluate()` always generates with `EVAL_SEED = 42`, so the
canonical measurement condition is fully reproducible and unambiguous.

## Canonical Measurement Condition
- Generator seed: `42` (`task.EVAL_SEED`)
- Sweep: `N ∈ {5, 10, 20, 40}`, 20 instances each
- Start city: `0`
- Headline / canonical problem size: `N = 10` (`task.CANONICAL_N`)
- Headline metric: `size_robustness` (NN accuracy retained from the smallest to
  the largest problem size)

## Model Function Signature
Attempts provide a `model_fn` implementing one decode step of an NN router:
```python
def model_fn(coords: np.ndarray,        # (N, 2) float, city coordinates
             current_idx: int,          # index of the current city
             visited: np.ndarray        # (N,) bool, True where already visited
            ) -> np.ndarray:            # (N,) float, one attention logit per city
    """
    Return one logit per city. The evaluator masks visited cities to -inf and
    takes the argmax, so a perfect mechanism places its maximum logit on the
    nearest unvisited city. Logits need not be normalised.
    """
```
The attempt's `main.py` wraps its model's forward pass in this function and
passes it to `task.evaluate()`. The smoke-test reference implementation is
`task.random_model_fn()` (emits random logits).

## Payload Contract
`task.evaluate(model_fn)` returns a dict with exactly these keys (matching
`benchmark.score` expectations):
```python
{
    "version": 1,                    # int, must match benchmark.VERSION
    "model_name": str,               # free-form label, metadata only
    "canonical_n": 10,               # int, the headline problem size
    "n_cities_sweep": [5, 10, 20, 40],   # list[int], the sweep axis
    "sweep": [                       # one record per problem size, same order
        {
            "n": int,                # problem size for this record
            "nn_accuracy": float,    # mean step-wise fraction of steps whose
                                     #   argmax == true nearest unvisited city
            "tour_length_ratio": float,  # mean(nn_tour_len / model_tour_len),
                                         #   clipped to [0, 1]; bigger = better
            "n_instances": int,      # instances aggregated into this record
        },
        ...
    ],
    "random_baseline": [             # same shape as `sweep`, random policy
        {"n": int, "nn_accuracy": float,
         "tour_length_ratio": float, "n_instances": int},
        ...
    ],
}
```
`sweep` and `random_baseline` are both length `len(n_cities_sweep)` and ordered
to match `n_cities_sweep`. All floats are native Python `float`.

`benchmark.score` **requires** `version`, `canonical_n`, `n_cities_sweep`,
`sweep`, `random_baseline`. `model_name` is optional metadata.

## Metrics
`benchmark.score(payload)` returns a flat dict of scalars:
| Metric | Formula / Meaning | Direction |
|--------|-------------------|-----------|
| `version` | Echo of `VERSION`. | — |
| `nn_accuracy_n_<N>` | Mean step-wise NN accuracy at problem size `N`. | Bigger = better |
| `tour_length_ratio_n_<N>` | Mean `nn_tour_len / model_tour_len` at size `N`, clipped `[0,1]`. | Bigger = better |
| `random_baseline_nn_accuracy_n_<N>` | Same accuracy metric for the random policy at size `N`. | Reference |
| `random_baseline_tour_length_ratio_n_<N>` | Tour-length ratio for the random policy at size `N`. | Reference |
| `nn_accuracy_canonical` | `nn_accuracy_n_<canonical_n>` (N=10). | Bigger = better |
| `tour_length_ratio_canonical` | `tour_length_ratio_n_<canonical_n>`. | Bigger = better |
| `lift_over_baseline_canonical` | `nn_accuracy_canonical - random_baseline_nn_accuracy_n_10`. Positive = beats random. | Bigger = better |
| `nn_accuracy_mean` | Mean of `nn_accuracy_n_<N>` over the sweep. | Bigger = better |
| `tour_length_ratio_mean` | Mean of `tour_length_ratio_n_<N>` over the sweep. | Bigger = better |
| `size_robustness` (**headline**) | `nn_accuracy` at the largest size ÷ at the smallest size, clipped `[0,1]`. `1.0` = scales perfectly, `→0` = falls apart as cities grow. `0.0` when the small-size accuracy is `≤ 1e-12`. | Bigger = better |

Per-slice keys use the `_n_<N>` suffix (integer city count). The headline
summary (`size_robustness`), all per-slice values, and the random baseline are
all present.

## Edge Cases
- `tour_length_ratio` (`task._safe_ratio`): returns `0.0` when the model tour
  length is `≤ 1e-12`, otherwise clips `nn_len / model_len` into `[0, 1]`.
- `nn_accuracy`: `0.0` when there are no decode steps (`steps == 0`).
- `size_robustness`: `0.0` when small-size accuracy is `≤ 1e-12` (avoids
  division by zero).
- Sweep aggregates divide by record counts only when non-empty, else `0.0`.

## Bump Procedure
- `VERSION` in `benchmark.py` increments on: any metric formula change, any
  payload key rename/removal/retype, or a change to the canonical condition
  (seed, sweep, canonical size, start city).
- The README metrics table and payload contract are updated in the same commit.
- Old `benchmark.json` files are retained; the dashboard filters to the highest
  version.

## GPU Requirement
`GPU_REQUIREMENT = 1`. The task/benchmark themselves are pure NumPy and run on
CPU; the pipeline still allocates one GPU slot for the attempt subprocess.

## Failure Detection
`is_obviously_broken(metrics)` returns `True` if:
- Any metric is NaN/Inf, or
- `nn_accuracy_canonical ≤ random_baseline_nn_accuracy_n_10` (the mechanism
  does not beat random city selection at the canonical condition).
