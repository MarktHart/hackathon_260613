# attention_global_align

## The question

When an attention head implements a **retrieval** function — a query position
that should attend to one specific key position — does it stay **globally
aligned** to the correct target across many different inputs, and how robust is
that alignment as an **interfering distractor key** is made progressively more
similar to the true target?

A head that puts all its mass on the right key when the distractor is
orthogonal, but splits or collapses onto the distractor as the distractor
approaches the target, is *not* robustly aligned. We quantify exactly how much
alignment survives maximum interference.

## Setup

**Synthetic**, fully deterministic given a seed. No trained model, no dataset,
no I/O.

Each retrieval problem is:

- a unit **query** vector `q` of dimension `d`;
- a **key matrix** `K` of shape `(L, d)` of unit key vectors;
- a designated **target** position `t` whose key equals `q` (cosine 1), i.e.
  the key a working head should retrieve;
- a **distractor** position `dpos` whose key has a *controlled cosine* `c` to
  the target key. As `c -> 1` the distractor becomes indistinguishable from the
  target and competes for attention.

We sweep `c` (the interference axis) and average over many sequences per slice.

| constant | value |
|----------|-------|
| `d` (dim)                 | 32 |
| `L` (key positions)       | 12 |
| `N_SEQS` (per slice)      | 24 |
| `distractor_cos_sweep`    | `[0.0, 0.25, 0.5, 0.75, 1.0]` |
| canonical distractor cos  | `0.5` |
| eval seed                 | `7` |

## Canonical measurement condition

Every attempt is evaluated by `task.evaluate`, which calls `generate(seed=7)`
and runs the attempt's `model_fn` over the full sweep above. The **canonical**
slice — the one reported as `*_canonical` — is `distractor_cos = 0.5`. Attempts
must not re-roll the data or change these constants.

## The `model_fn` contract

The attempt hands `evaluate` a single callable:

```python
model_fn(q: np.ndarray, K: np.ndarray) -> np.ndarray
#   q : shape (d,)      unit query vector
#   K : shape (L, d)    key matrix
# returns: shape (L,)   attention *logits* over the L key positions
```

Return **logits**, not a normalised distribution — `evaluate` applies the
softmax over the L keys itself, so any finite real vector of length `L` is
valid. The natural "correct" mechanism is `K @ q`. Logits that are non-finite
or of the wrong shape raise `ValueError`.

`task.random_model_fn()` returns a reference `ModelFn` (random logits) used by
the pipeline smoke test; it takes **no arguments** and returns a callable with
the signature above.

## Payload contract

`task.evaluate` returns, and `benchmark.score` consumes, exactly:

| key | type | semantics |
|-----|------|-----------|
| `version`                  | int   | payload schema version (== 1) |
| `model_name`               | str   | label, not scored |
| `d`                        | int   | probe dimension |
| `seq_len`                  | int   | number of key positions `L` |
| `canonical_distractor_cos` | float | canonical slice (0.5) |
| `distractor_cos_sweep`     | list[float] | interference axis values |
| `sweep`                    | list[record] | one per sweep value |
| `uniform_baseline`         | list[record] | one per sweep value |

Each `sweep` record (mean over `N_SEQS` sequences):

| field | type | semantics |
|-------|------|-----------|
| `distractor_cos`   | float | the slice value |
| `global_alignment` | float | mean attention mass on the **target** key, `[0,1]` |
| `distractor_mass`  | float | mean attention mass on the **distractor** key, `[0,1]` |
| `target_margin`    | float | mean `target_mass - distractor_mass`, `[-1,1]` |
| `n_seqs`           | int   | sequences averaged |

Each `uniform_baseline` record: `{distractor_cos, global_alignment, n_seqs}`,
where `global_alignment = 1/L` (mass a uniform head places on the target).

## Metrics

Returned by `benchmark.score`. `version` is always first; the dashboard filters
to the highest version present. **Bigger is better** for every metric here.

| metric | meaning |
|--------|---------|
| `global_alignment_robustness` | **Headline.** `alignment(max interference) / alignment(no interference)`, clipped to `[0,1]`. 1.0 = no alignment lost as the distractor approaches the target. |
| `global_alignment_canonical`  | alignment at the canonical slice (`cos = 0.5`) |
| `lift_over_uniform_canonical` | canonical alignment minus the uniform baseline |
| `global_alignment_mean`       | alignment averaged over the whole sweep |
| `global_alignment_dist_<c>`   | per-slice alignment (`_dist_0p0` … `_dist_1p0`) |
| `distractor_mass_dist_<c>`    | per-slice mass on the distractor |
| `target_margin_dist_<c>`      | per-slice target-minus-distractor margin |
| `uniform_baseline_alignment_dist_<c>` | per-slice uniform reference (`1/L`) |

Slice floats use `0p25`-form, not `0.25`.

### Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU; `task`/`benchmark` stay CPU.
- `is_obviously_broken(metrics)` — `True` (skip jury) on any NaN/inf, or when
  `global_alignment_canonical <= uniform_baseline_alignment_dist_0p5` (the head
  failed to beat a uniform head and has no mechanism worth judging).

## Bump procedure

`VERSION` lives in `benchmark.py`. Bump it (and update this contract in the same
commit) when you change any existing metric formula, rename/retype/remove a
payload key, or move the canonical condition. Adding a new metric, an optional
payload key, or an extra sweep slice does **not** require a bump. Old
`benchmark.json` files stay on disk; the dashboard hides superseded versions.
