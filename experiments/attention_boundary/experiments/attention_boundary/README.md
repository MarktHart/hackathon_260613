# Attention Boundary Respect

## Question

When an interpretability method reconstructs the attention pattern of a small
model, does that pattern **respect segment boundaries** — i.e. do query tokens
concentrate their attention *within their own segment* rather than leaking
across a delimiter into the other segment? A method that recovers a genuine
boundary-aware mechanism should keep attention mass inside the querying token's
segment; a method that merely returns smeared / uniform attention will not.

## Setup

**Synthetic generator only.** Each sequence has a fully fixed two-segment
structure with a known boundary, so the ground-truth "within / cross / delim /
eos" partition of every key is exact — no Torch, no GPU, deterministic.

Sequence layout (length `seq_len = 18`):

```
[ segA: 8 tokens ][ DELIM ][ segB: 8 tokens ][ EOS ]
   indices 0..7      8         9..16            17
```

- Vocabulary size `64`. `DELIM = 63`, `EOS = 62`.
- `segA` tokens are sampled from `1..31`; `segB` tokens from `32..61`.
- The **structure is fixed**; `generate`'s `seed` only changes *which* tokens
  are sampled inside each segment. Same seed → same batch (verified
  deterministic).

**Canonical measurement condition**

- `vocab_size = 64`
- `seg_len = 8`
- `batch_size = 32`
- `n_heads = 4`
- `seq_len = 18`
- `delim_pos = 8`
- `canonical_seed = 0` (`task.evaluate` calls `generate(seed=0)`)

## Model-function contract

Every attempt must provide a `model_fn` with this exact signature:

```python
def model_fn(input_ids: np.ndarray, delim_id: int) -> np.ndarray:
    """
    Args:
        input_ids: np.ndarray[int32], shape (batch, seq_len). The canonical
                   batch produced by task.generate(seed=0).
        delim_id:  int, the delimiter token id (63 in the canonical config).

    Returns:
        np.ndarray[float], shape (batch, n_heads, seq_len, seq_len) — per-head
        attention weights. The LAST axis (keys) must be a probability
        distribution: it sums to 1 (checked with atol=1e-3).
    """
```

`task.evaluate` validates both the shape and the row-normalisation and raises
`ValueError` if either is violated. The attempt's `main.py` should `import`
`generate`/`evaluate` from `task.py` and pass its own `model_fn`. The
framework's smoke test calls `task.evaluate(task.random_model_fn())`;
`random_model_fn` returns a `model_fn` that emits a valid **uniform** attention
distribution of the correct shape (the no-mechanism reference).

## Payload contract

`task.evaluate` returns a `dict` with this exact structure (Python types shown):

```python
{
    "version": 1,                          # int, matches benchmark.VERSION
    "config": {                            # dict, frozen run configuration
        "vocab_size": 64,
        "seg_len": 8,
        "batch_size": 32,
        "n_heads": 4,
        "seq_len": 18,
        "delim_pos": 8,
        "canonical_seed": 0
    },
    "sweep": [                             # list[dict], exactly 2 records
        {
            "query_segment": "segA",       # str, "segA" or "segB"
            "within_seg_attn": 0.0,        # float, mean mass to own segment's keys
            "delim_attn": 0.0,             # float, mean mass to the DELIM key
            "cross_seg_attn": 0.0,         # float, mean mass to the other segment
            "eos_attn": 0.0,               # float, mean mass to the EOS key
            "head_sharpness": [0.0, 0.0, 0.0, 0.0]  # list[float], length n_heads
        },
        { "query_segment": "segB", ... }   # same keys
    ],
    "linear_baseline": {                   # dict, uniform-attention reference
        "segA": { ...same fields as a sweep record minus query_segment... },
        "segB": { ... }
    }
}
```

**Region semantics.** For the query positions of a segment, each of the four
region masses is the attention mass summed over that region's key positions,
then averaged over batch × heads × query positions. The four regions (`within`,
`delim`, `cross`, `eos`) partition *all* keys, so
`within + delim + cross + eos == 1` up to float error.

**Per-head sharpness.** For each head, `head_sharpness[h] = head_within[h] -
max(head_delim[h], head_cross[h], head_eos[h])` — how much more mass a head
sends within-segment than to its best competing region. A perfectly
boundary-respecting head → `1.0`; uniform attention → `0.0`.

`sweep` is a list so future multi-condition variants (e.g. varying `seg_len`)
can append records without a contract change.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| Metric | Formula | Direction | Notes |
|--------|---------|-----------|-------|
| `version` | `payload["version"]` | — | First key, dashboard filter |
| `boundary_sharpness_canonical` | mean over {segA, segB} of mean-over-heads `head_sharpness` | **Bigger = better** | Headline summary |
| `boundary_sharpness_segA` | mean-over-heads `head_sharpness` of segA record | Bigger = better | Per-slice |
| `boundary_sharpness_segB` | mean-over-heads `head_sharpness` of segB record | Bigger = better | Per-slice |
| `boundary_crossing_rate_canonical` | mean over {segA, segB} of `cross_seg_attn` | Smaller = better | Leakage to the other segment |
| `boundary_crossing_rate_segA` | `cross_seg_attn` of segA record | Smaller = better | Per-slice |
| `boundary_crossing_rate_segB` | `cross_seg_attn` of segB record | Smaller = better | Per-slice |
| `delim_leakage_canonical` | mean over {segA, segB} of `delim_attn` | Smaller = better | Mass parked on the delimiter |
| `delim_leakage_segA` | `delim_attn` of segA record | Smaller = better | Per-slice |
| `delim_leakage_segB` | `delim_attn` of segB record | Smaller = better | Per-slice |
| `linear_baseline_sharpness_canonical` | same formula over `linear_baseline` | Bigger = better | Reference (uniform ⇒ 0) |
| `linear_baseline_sharpness_segA` | baseline segA sharpness | Bigger = better | Reference |
| `linear_baseline_sharpness_segB` | baseline segB sharpness | Bigger = better | Reference |
| `lift_over_linear_sharpness` | `boundary_sharpness_canonical - linear_baseline_sharpness_canonical` | Bigger = better | Improvement over strawman |

The headline metric is **`boundary_sharpness_canonical`**. `crossing_rate` and
`delim_leakage` are *leakage* metrics — their names denote the thing you want
small, so smaller is better; everything else is bigger-is-better.

**Linear (no-mechanism) baseline** (computed inside `task.evaluate`): uniform
attention over all `seq_len` keys. Region masses are then proportional to region
size — `within = cross = seg_len/seq_len = 8/18 ≈ 0.444`, `delim = eos =
1/18 ≈ 0.056` — and every head's `head_sharpness` is `0`, so baseline sharpness
is `0`.

**Edge cases.** `score()` averages with a guarded mean (empty list → `0.0`),
validates that every region field is finite (rejects NaN/Inf with `ValueError`),
and requires a non-empty `head_sharpness` list per record. There are no raw
divisions that can hit a zero denominator.

**`is_obviously_broken`** returns `True` (skips the jury) if:
- any metric is NaN/Inf, or
- `boundary_sharpness_canonical <= max(linear_baseline_sharpness_canonical, 0) + 0.05`
  (fails to clear the uniform baseline by a clear margin), or
- `boundary_crossing_rate_canonical > 0.5` (more mass to the other segment than
  a uniform read would give).

## Bump procedure

Bump `VERSION` in `benchmark.py` and update this README in the same commit when:
- Any metric formula changes.
- Payload keys are added, removed, or retyped.
- Canonical condition parameters change (`seg_len`, `seq_len`, `batch_size`,
  `n_heads`, `canonical_seed`, etc.).

Do **not** bump for adding a new metric without changing existing ones, adding
an optional payload key with a default, or appending a new slice to `sweep`.
