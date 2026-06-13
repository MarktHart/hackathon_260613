# attention_argmin

## Question

Does an attention head implement an **argmin** over per-position values — i.e.
does the attention distribution concentrate its mass on the position holding
the *minimum* value in the sequence, and how sharply does it do so relative to
a no-mechanism (uniform) reference as the task gets harder?

## Setup

**Synthetic generator** (pure NumPy, no GPU). Each sequence has `seq_len = 64`
positions. Every position carries a key vector in ℝ³² and a scalar value. A
single fixed query is used for all sequences. The ground-truth target is the
index of the minimum value.

Difficulty is controlled by the **gap** — the margin between the minimum and
the runner-up value:

- distractor values `values[i] ~ Uniform(-1, 1)`;
- the minimum is set to `-1 - gap` at a random position;
- the runner-up is set to `-1 + gap` at another random position.

Small gaps make the argmin hard to resolve; large gaps make it easy. Keys are
random unit vectors (shared across sequences); the query is `[1, 0, …, 0]`.

## Canonical measurement condition

| Parameter | Value |
|-----------|-------|
| Sequence length | 64 |
| Key dimension | 32 |
| Gap sweep | `[0.05, 0.1, 0.2, 0.5, 1.0, 2.0]` |
| **Canonical gap** | `0.5` |
| Sequences per gap | 200 |
| Seed | `0` (fixed for all attempts) |

`generate(seed)` is deterministic: same seed → identical `Batch`. `evaluate`
always uses `seed = 0`, the canonical condition; passing a different seed to
`generate` is supported but not used by the scored pipeline.

## Model function signature

Attempts provide a callable with **exactly** this signature (one sequence at a
time):

```python
def model_fn(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    """
    keys:   (seq_len, key_dim) float32
    values: (seq_len,)         float32
    query:  (key_dim,)         float32
    returns: (seq_len,) attention weights — non-negative, summing to ~1.
    """
```

`task.random_model_fn()` returns a shape-correct dummy (uniform attention) used
by the pipeline smoke test.

## Payload contract

`task.evaluate(model_fn)` returns a dict with **exactly** these keys:

```python
{
    "version": 1,
    "canonical": <record>,              # the record whose gap == 0.5
    "sweep": [<record>, ...],           # one record per gap, in GAPS order
    "linear_baseline": {                # uniform-attention strawman, same batch
        "canonical": <record>,
        "sweep": [<record>, ...],       # same length / order as "sweep"
    },
    "model_config": {                   # self-describing metadata
        "seq_len": 64, "key_dim": 32,
        "n_seq_per_gap": 200,
        "gaps": [0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
        "canonical_gap": 0.5, "seed": 0,
    },
}
```

Each `<record>` is:

| Field | Type | Meaning |
|-------|------|---------|
| `gap` | `float` | The margin for this slice. |
| `sequences` | `int` | Number of sequences averaged (200). |
| `seq_len` | `int` | Positions per sequence (64); the uniform per-position share is `1/seq_len`. |
| `attn_at_min` | `float` | Mean attention weight on the true argmin position. |
| `attn_at_others` | `float` | Mean attention weight **per non-argmin position** (`(1 - attn_at_min)/(seq_len-1)` for a normalised head). Diagnostic; not used by the sharpness metric. |
| `argmax_correct` | `float` | Fraction of sequences where `argmax(attn)` equals the true argmin. |

All floats are Python `float`. The `linear_baseline` records have the same
shape and are produced by uniform attention over the identical batch, so
sharpness contrasts are apples-to-apples.

## Metrics

`benchmark.score(payload)` returns a flat dict. Sharpness is the core
concentration measure — attention on the argmin relative to the uniform
per-position share `1/seq_len`:

```
sharpness(record) = attn_at_min / (1 / seq_len) = attn_at_min * seq_len
                                                  # guarded: 0.0 if seq_len <= 0
                                                  # or attn_at_min non-finite
```

Uniform attention gives `sharpness == 1.0`; a perfect argmin head (all mass on
the true minimum) gives `sharpness == seq_len` (`64`), the maximum. The measure
is bounded in `[0, seq_len]` and well-defined at the optimum — a ratio against
`attn_at_others` would instead divide by zero exactly when the head is perfect.
Gap tags use two-decimal `0p50` form.

| Metric | Formula | Direction |
|--------|---------|-----------|
| `version` | `payload["version"]` | — (always first key) |
| `argmin_sharpness_canonical` | `sharpness(canonical)` | **bigger better** — headline |
| `argmin_accuracy_canonical` | `canonical["argmax_correct"]` | bigger better |
| `argmin_attn_at_min_canonical` | `canonical["attn_at_min"]` | bigger better |
| `linear_baseline_sharpness_canonical` | `sharpness(baseline canonical)` | reference (≈ 1.0) |
| `linear_baseline_accuracy_canonical` | baseline `argmax_correct` | reference |
| `lift_over_baseline_canonical` | `argmin_sharpness_canonical - linear_baseline_sharpness_canonical` | bigger better |
| `argmin_sharpness_gap_0p05` … `_2p00` | per-slice sharpness | bigger better |
| `argmin_accuracy_gap_0p05` … `_2p00` | per-slice `argmax_correct` | bigger better |
| `linear_baseline_sharpness_gap_*` | per-slice baseline sharpness | reference |
| `lift_over_baseline_gap_*` | per-slice `sharpness - baseline` | bigger better |
| `argmin_robustness` | `min(sharpness over sweep) / max(sharpness over sweep)` | bigger better, ∈ [0, 1] |
| `worst_slice_sharpness` | `min(sharpness over sweep)` | bigger better |

**Headline metric:** `argmin_sharpness_canonical`.

### Edge cases

- Any sharpness with `seq_len <= 0` or non-finite `attn_at_min` → `0.0`.
- Empty `sweep` or a `linear_baseline.sweep` of mismatched length → `ValueError`.
- Missing required keys → `KeyError` with a descriptive message.
- `is_obviously_broken` short-circuits the jury when any metric is NaN/inf, or
  when the canonical sharpness does not exceed the uniform baseline.

## Bump procedure

Increment `VERSION` in `benchmark.py` (and update this README in the same
commit) when:

- any metric formula changes;
- a payload key is added/removed/retyped, or a record field changes;
- the canonical condition changes (gap, sequence count, dimensions, seed).

Do **not** bump when adding a new metric that leaves existing ones untouched,
or adding an optional payload key with a default. Old `benchmark.json` files
stay on disk; the dashboard filters to the highest version present.
