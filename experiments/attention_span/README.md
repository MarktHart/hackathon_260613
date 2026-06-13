# attention_span

## Question

What is the effective **span** of a transformer's attention mechanism — over
what range of distances can it maintain focused attention on a relevant
(target) token?

## Setup

**Synthetic needle-in-haystack generator.**
Fixed sequence length `L = 512`. Every sequence is filled with random filler
tokens (integers in `[1, 1000)`). A *query* token (value `8888`) is written at
position `0`. A *target* ("needle") token (value `9999`) is written at a
varying distance `d` from the query. Attention is measured **from the query
position (0) to the target position (d)**, so `distance == d`.

The sweep covers `d ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256}` with **100
sequences per distance** → a batch of **900** sequences. Per-distance values
are means over those 100 samples.

`task.generate(seed)` is deterministic: the same seed produces the same batch.
The filler tokens are drawn from `np.random.default_rng(seed)`; the canonical
seed is `0`. The needle/query positions and the sweep distances are fixed and
do **not** depend on the seed.

The attempt provides a `model_fn` that, given the batch of token ids, returns
the attention tensor. This is model-agnostic: the function can extract
attention from a real model, run a probe, or compute a heuristic — anything
that yields a per-(query, key) attention matrix.

## Canonical measurement condition

| Parameter | Value |
|-----------|-------|
| Sequence length | 512 |
| Batch size | 900 |
| Samples per distance | 100 |
| Query token | `8888` (at position 0) |
| Target/needle token | `9999` (at position `d`) |
| Filler tokens | random ints in `[1, 1000)` |
| Sweep distances | `[1, 2, 4, 8, 16, 32, 64, 128, 256]` |
| Seed | `0` (canonical; fillers depend on seed, geometry does not) |

Every attempt **must** evaluate on this batch via `task.generate(seed=0)`,
which `task.evaluate` calls internally.

## `model_fn` signature

```python
ModelFn = Callable[[np.ndarray], np.ndarray]
# input_ids (batch, seq_len) int32 -> attention weights
```

- Input: `input_ids`, a 2-D `np.ndarray` of shape `(batch, seq_len)`, dtype
  `int32`.
- Output: an attention array of shape **`(batch, num_heads, seq_len, seq_len)`**
  or **`(batch, seq_len, seq_len)`**. Entry `[b, h, i, j]` (or `[b, i, j]`) is
  the attention weight from query position `i` to key position `j` in
  sequence `b`. A 3-D return is treated as a single head.
- `evaluate` averages over heads, then reads the `(query_position=0,
  target_position=d)` entry for each sequence. Returns of any other
  dimensionality, or with a mismatched batch/sequence length, raise
  `ValueError`.

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with the following exact keys:

```python
{
    "version": 1,                       # int — payload version
    "canonical_seq_len": 512,           # int
    "canonical_distances": [1, 2, 4, 8, 16, 32, 64, 128, 256],  # list[int]
    "samples_per_distance": 100,        # int
    "sweep": [                          # list[dict], one per distance (length 9)
        {
            "distance": 1,                          # int
            "mean_attention_on_target": 0.85,       # float — mean over 100 samples
            "std_attention_on_target": 0.01,        # float — std over 100 samples
            "n_samples": 100,                       # int
        },
        ...
    ],
    "attention_span_auc": 0.42,         # float — log2-normalised AUC (see Metrics)

    # Attempts MAY add extra keys (e.g. "model_name"); score() ignores them.
}
```

- `sweep` is ordered by increasing `distance` (canonical order).
- `mean_attention_on_target` / `std_attention_on_target` are raw attention
  weights (model outputs), aggregated over the 100 samples at that distance.
- The payload is **pre-aggregated**: per-sequence reduction happens inside
  `evaluate`; `score()` consumes only the per-distance scalars.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`. `version`
is emitted first. For the attention-span metrics, **bigger is better** (more
attention mass retained over distance).

### Headline summary

| Key | Formula | Interpretation |
|-----|---------|----------------|
| `attention_span_auc_canonical` | Trapezoidal AUC of `mean_attention_on_target` vs `log2(distance)`, normalised by the `log2`-distance range. Equals the distance-weighted average attention weight in `[0, 1]`. | Total attention mass on the target across distances. Higher = wider *and* stronger span. Recomputed in `score()` from the sweep (mirrors the payload's `attention_span_auc`). |

### Per-slice values

| Key pattern | Example |
|-------------|---------|
| `attention_on_target_dist_<d>` | `attention_on_target_dist_16` = `mean_attention_on_target` at distance 16 (one key per sweep distance). |

### Secondary metrics & baselines

| Key | Formula | Note |
|-----|---------|------|
| `attention_span_robustness` | `mean_attention_on_target[max_d] / mean_attention_on_target[min_d]` | Ratio of long-range to short-range attention. `0.0` if the short-range value is `0`. |
| `linear_baseline_attention_span_auc` | `1 / canonical_seq_len` | Uniform-attention reference (`1/512`), same units as the headline AUC. |
| `linear_baseline_attention_on_target_dist_1` | `1 / canonical_seq_len` | Uniform-attention reference at the nearest distance. |
| `lift_over_baseline_auc` | `attention_span_auc_canonical − 1 / canonical_seq_len` | Headline AUC minus the uniform baseline, same units. |

### Validation & edge cases

`score()` is defensive:

- Raises `KeyError` if any required key (`version`, `canonical_seq_len`,
  `canonical_distances`, `samples_per_distance`, `sweep`,
  `attention_span_auc`) is missing.
- Raises `ValueError` on unsupported `version` or an empty `sweep`.
- The AUC denominator (`log2`-range) is guarded: a single-point sweep yields
  AUC `0.0` rather than dividing by zero.
- `attention_span_robustness` returns `0.0` when the min-distance attention is
  `0` (no division by zero).

### Pipeline hooks

- `GPU_REQUIREMENT = 1` — every attempt runs on the GPU.
- `is_obviously_broken(metrics)` — returns `True` (skips the jury) if any
  metric is `NaN`/`inf`, or if `attention_span_auc_canonical <=
  linear_baseline_attention_span_auc * 1.01` (no meaningful lift over uniform
  attention).

## Version bumping

`VERSION` in `benchmark.py` (currently `1`) must be incremented when:

- Any metric formula changes.
- Payload keys are renamed, removed, or retyped.
- The canonical condition changes (sequence length, sweep distances,
  samples-per-distance, token values).

Adding new metrics or optional payload keys does **not** require a bump.

## Bump procedure

1. Increment `VERSION` in `benchmark.py`.
2. Update the "Payload contract" and "Metrics" tables in this README in the
   same commit.
3. Old `benchmark.json` files remain on disk; the dashboard filters to the
   highest version.
