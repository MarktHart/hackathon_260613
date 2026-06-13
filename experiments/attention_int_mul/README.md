# attention_int_mul

## Question

Can a single attention head implement **integer multiplication routing**?
Given a query that encodes two operands `a` and `b`, and a row of candidate
positions each encoding an integer value, does the head attend **only** to the
position whose value equals the product `a × b` — and does it keep doing so as
the operand range grows (more, and more confusable, distractor products)?

Multiplication is the interesting mechanism here: a head that scores keys by a
*linear* (additive) match of the operands can route by `a + b` or by either
operand alone, but cannot single out `a × b`. The goal measures whether an
attempt's head genuinely multiplies.

## Setup

**Synthetic generator** — fully controlled, no trained datasets. Every integer
`n ∈ {0 … V-1}` (with `V = 1024`) has a fixed unit embedding `φ(n) ∈ ℝ^d`
(`d = 128`), produced once from a constant seed and **stable across all
batches** so an attempt can learn or reconstruct the map.

Each routing trial:

- sample operands `a, b ∈ {0 … K-1}`; the ground-truth product is `p = a·b`;
- build `n_positions = 16` candidate integer values: the product `p` plus
  `n_positions − 1` distractors, drawn preferentially from *other valid
  products* of the same `K` (so they are confusable), padded with uniform
  integers from `[0, V-1]` when a small `K` lacks enough distinct products;
- the candidate row is shuffled; exactly one column holds `p`.

The head receives `φ(a)`, `φ(b)`, and the `n_positions × d` matrix of candidate
embeddings, and must attend to the product column.

We sweep the operand range `K`, which widens the product space and packs the
distractor products closer together.

### Canonical measurement condition

- `d = 128` (embedding dimension), `V = 1024` (integer-embedding table)
- `n_positions = 16` candidates per trial
- sweep: `K ∈ {2, 4, 8, 16, 32}` (operands drawn from `[0, K-1]`)
- canonical `K = 8`
- `N_TRIALS = 200` trials per `K`, averaged
- evaluation batch uses a fixed seed (`generate(seed=42)`); `generate` is
  deterministic for any given seed. The integer embedding `φ` is fixed
  independently of the batch seed.

## Model function signature

The goal's contract with attempts. An attempt provides a `model_fn` and hands
it to `task.evaluate`; it never builds the payload itself.

```python
def model_fn(a_vec: np.ndarray, b_vec: np.ndarray, key_vecs: np.ndarray) -> np.ndarray:
    """
    Args:
        a_vec:    (d,)               embedding phi(a) of operand a
        b_vec:    (d,)               embedding phi(b) of operand b
        key_vecs: (n_positions, d)   embeddings phi(value_i) of each candidate

    Returns:
        attn_logits: (n_positions,)  unnormalised attention logits
                     (higher = more attention)
    """
```

The attempt returns raw logits; `task.evaluate` applies softmax and computes
all metrics. The fixed embedding is exposed as `task.embed(n)` / `task.INT_EMBED`
so attempts can decode integers. `task.random_model_fn()` returns a reference
`model_fn` that emits random logits of the correct shape (used by the smoke
test) and routes at chance.

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                       # int, matches benchmark.VERSION
    "model_name": "synthetic_attention_int_mul",
    "d": 128,                           # int, embedding dimension
    "n_positions": 16,                  # int, candidates per trial
    "canonical_k": 8,                   # int, the canonical condition
    "k_sweep": [2, 4, 8, 16, 32],       # list[int], the sweep axis
    "sweep": [                          # one record per k_sweep value
        {
            "k": 8,                     # int, operand range
            "routing_accuracy": 0.91,   # float in [0,1], argmax == product col
            "attended_mass": 0.78,      # float in [0,1], softmax mass on product
            "n_trials": 200,            # int
        },
        ...
    ],
    "linear_baseline": [                # same axis, no-mechanism (additive) ref
        {
            "k": 8,                     # int
            "routing_accuracy": 0.09,   # float in [0,1]
            "n_trials": 200,            # int
        },
        ...
    ],
}
```

`sweep` and `linear_baseline` are both lists of the same length as `k_sweep`,
each indexed by its `k` field. All accuracy and mass values are in `[0, 1]`;
higher is better.

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars:

| metric | meaning | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` (= 1) | — |
| `routing_accuracy_k_2` … `routing_accuracy_k_32` | per-K routing accuracy | **bigger = better** |
| `attended_mass_k_2` … `_k_32` | per-K softmax mass on the product column | **bigger = better** |
| `linear_baseline_accuracy_k_2` … `_k_32` | additive-baseline accuracy per K | reference |
| `routing_accuracy_canonical` | accuracy at `canonical_k` (8) | **bigger = better** |
| `attended_mass_canonical` | attended mass at `canonical_k` | bigger = better |
| `linear_baseline_accuracy_canonical` | baseline accuracy at `canonical_k` | reference |
| `lift_over_baseline_canonical` | `routing_accuracy_canonical − linear_baseline_accuracy_canonical` | bigger = better |
| `mean_routing_accuracy` | mean routing accuracy across the sweep | **bigger = better** (headline) |
| `mean_linear_baseline_accuracy` | mean baseline accuracy across the sweep | reference |
| `routing_robustness` | accuracy at max K (32) ÷ accuracy at min K (2), clipped `[0,1]` | bigger = better |

### Headline summary

**`mean_routing_accuracy`** — the average fraction of trials, across the
operand-range sweep, where the head attends to the product position. Chance is
`1 / n_positions ≈ 0.0625`; a head that genuinely multiplies approaches `1.0`.
The additive baseline sits well *above* chance at small `K` (≈ `0.5` at
`K = 2`, decaying as `K` grows — it succeeds whenever the product happens to
coincide with an operand, e.g. `×0`, `×1`, or a square) but cannot single out
`a × b` in general; it is the no-mechanism reference, so the meaningful signal
is `lift_over_baseline_canonical` and the per-`K` gap, not the absolute number.
(`routing_robustness` is a secondary
view of how gracefully accuracy degrades as `K` grows; an absolute headline is
used because a ratio is near `1.0` for a chance-level model and would not rank
attempts sensibly.)

## Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU (the smoke test runs
  `task`/`benchmark` on CPU/NumPy).
- `is_obviously_broken(metrics)` — short-circuits the jury when metrics are
  NaN/inf or fail to beat the additive baseline on `mean_routing_accuracy`.

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README, same commit) when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- `canonical_k`, the sweep values, `n_positions`, `d`, `V`, or the embedding
  scheme change;
- a sweep record's schema changes.

Do **not** bump when adding a new metric that leaves existing ones unchanged,
or adding an optional payload key with a default. This goal is at `VERSION = 1`.
Old `benchmark.json` files of a lower version stay on disk; the dashboard
filters to the highest version present.
