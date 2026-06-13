# attention_and

## Question

Can a single attention head implement a clean **logical AND** over two
independent query directions (q_A and q_B) in superposition? Specifically,
does the head attend sharply **only when both** query features are present,
and suppress attention when either is absent — and does it keep doing so as
the two feature directions overlap (cosine grows from 0 toward 1)?

## Setup

**Synthetic generator** — fully controlled, no trained models. We construct a
minimal residual stream where two binary features (A, B) are encoded as
orthogonal (or near-orthogonal) directions q_A, q_B ∈ ℝ^d. The ground-truth
attention pattern is:

```
Attend to position i  ⇔  feature A present at i  AND  feature B present at i
```

We sweep the cosine similarity cos(q_A, q_B) ∈ [0, 1] to test robustness to
superposition (non-orthogonal features).

### Canonical measurement condition

- `d = 64` (residual dimension)
- `n_positions = 100`
- feature density = `0.3` (each feature independently present at a position
  with probability 0.3)
- canonical cosine = `0.0` (orthogonal features)
- sweep: `cos ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}`
- `10` random seeds per cosine value, averaged
- evaluation batch uses a fixed seed (`generate(seed=42)`); `generate` is
  deterministic for any given seed.

## Model function signature

The goal's contract with attempts. An attempt provides a `model_fn` and hands
it to `task.evaluate`; it never builds the payload itself.

```python
def model_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """
    Args:
        q_A:      (d,)              query direction for feature A
        q_B:      (d,)              query direction for feature B
        residual: (n_positions, d) residual stream at each position

    Returns:
        attn_logits: (n_positions,) unnormalised attention logits
                     (higher = more attention)
    """
```

The attempt returns raw logits; `task.evaluate` applies softmax and computes
all metrics. `task.random_model_fn()` returns a reference `model_fn` that
emits random logits of the correct shape (used by the smoke test).

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 2,                       # int, matches benchmark.VERSION
    "model_name": "synthetic_attention_and",
    "d": 64,                            # int, residual dimension
    "canonical_cosine": 0.0,           # float, the canonical condition
    "cos_AB_sweep": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],   # list[float], the sweep axis
    "sweep": [                          # one record per cos_AB_sweep value
        {
            "cosine": 0.0,              # float, cos(q_A, q_B)
            "and_sharpness": 0.87,      # float in [0,1], AND-boundary sharpness
            "false_positive_rate": 0.03,# float in [0,1], attends when ¬(A∧B)
            "false_negative_rate": 0.08,# float in [0,1], misses when A∧B
            "n_seeds": 10,              # int
        },
        ...
    ],
    "linear_baseline": [                # same axis, no-mechanism reference
        {
            "cosine": 0.0,             # float
            "and_sharpness": 0.42,     # float in [0,1]
            "n_seeds": 10,             # int
        },
        ...
    ],
}
```

`sweep` and `linear_baseline` are both lists of the same length as
`cos_AB_sweep`, each indexed by its `cosine` field. All sharpness and rate
values are in `[0, 1]`; higher sharpness is better, lower FPR/FNR is better.

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars (floats `0.7` named
`0p7`):

| metric | meaning | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` (= 2) | — |
| `and_sharpness_cos_0p0` … `and_sharpness_cos_1p0` | per-cosine AND sharpness | **bigger = better** |
| `false_positive_rate_cos_0p0` … `_cos_1p0` | per-cosine FPR | smaller = better |
| `false_negative_rate_cos_0p0` … `_cos_1p0` | per-cosine FNR | smaller = better |
| `linear_baseline_sharpness_cos_0p0` … `_cos_1p0` | baseline sharpness per cosine | reference |
| `and_sharpness_canonical` | sharpness at `canonical_cosine` (0.0) | **bigger = better** |
| `lift_over_baseline_canonical` | `and_sharpness_canonical − linear_baseline_sharpness_cos_0p0` | bigger = better |
| `superposition_robustness` | sharpness at max cos (1.0) ÷ sharpness at min cos (0.0), clipped to `[0,1]` | **bigger = better** (headline) |

### Headline summary

**`superposition_robustness`** — the fraction of orthogonal-case sharpness
that survives at maximum superposition (cos = 1.0). A head that degrades
gracefully scores near 1.0; one that collapses when features overlap scores
near 0.0.

## Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU (the smoke test runs
  `task`/`benchmark` on CPU/NumPy).
- `is_obviously_broken(metrics)` — short-circuits the jury when metrics are
  NaN/inf or fail to beat the linear baseline at the canonical condition.
  (The gate is additive, not multiplicative: `and_sharpness` is clipped to
  `[0, 1]`, so requiring a multiple of a baseline that already sits near `0.77`
  would be unsatisfiable even for a perfect attempt.)

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README, same commit) when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- `canonical_cosine` or the sweep values change;
- a sweep record's schema changes.

Do **not** bump when adding a new metric that leaves existing ones unchanged,
or adding an optional payload key with a default. This goal is at `VERSION = 2`
(v1 measured only at the orthogonal anchor); old v1 `benchmark.json` files stay
on disk but the dashboard filters to the highest version present.
