# attention_local_align

## Question

Do attention heads in a model reliably attend to **local syntactic dependencies** — e.g. each token attending to its immediate predecessor (or successor) — when the data-generating process has exactly that structure?

## Setup

**Synthetic generator.** We construct sequences where every position `t > 0` has a *single* correct local alignment target: the previous token `t-1`. The ground-truth attention pattern is therefore a strict sub-diagonal (or super-diagonal for right-to-left) band of width 1.

- Vocabulary: 64 random tokens, no special structure.
- Sequence length: 32 (canonical).
- Batch size: 128 sequences per evaluation.
- The generator is fully deterministic given `seed`; `seed` is accepted but only used to shuffle the vocabulary once per seed.

**Model function contract.** Attempts provide a callable

```python
ModelFn = Callable[[np.ndarray], np.ndarray]
# input:  (B, T) int32 token ids
# output: (B, H, T, T) float32 attention weights, already softmax-normalised per query
```

Only head `head_idx = 0` is measured. The canonical condition uses `layer = 0`, `head = 0` of whatever model the attempt wraps.

## Canonical measurement condition

| Parameter          | Value |
|--------------------|-------|
| Sequence length    | 32    |
| Vocabulary size    | 64    |
| Batch size         | 128   |
| Measured head      | 0     |
| Sweep axis         | `shift` ∈ {-2, -1, 0, +1, +2} |
| Canonical shift    | -1 (immediate predecessor) |

The sweep varies the *ground-truth* alignment target by an integer `shift`:
- `shift = -1` → target is `t-1` (canonical, predecessor)
- `shift = +1` → target is `t+1` (successor)
- `shift = 0`  → target is `t`   (self)
- `shift = -2` → target is `t-2` (two back)
- `shift = +2` → target is `t+2` (two forward)

For each shift we generate a fresh batch with that ground-truth pattern and measure the model's alignment to it.

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,
    "canonical_shift": -1,
    "sequence_length": 32,
    "vocab_size": 64,
    "batch_size": 128,
    "measured_head": 0,
    "sweep": [
        {
            "shift": int,                    # -2, -1, 0, 1, 2
            "mean_max_attn_to_target": float, # mean over batch of max attention weight placed on the true target token
            "mean_entropy": float,           # mean attention entropy per query position (lower = sharper)
            "frac_peak_on_target": float,    # fraction of queries where the target token receives the maximum attention
        },
        ...
    ],
}
```

All floats are Python `float` (not `np.float32`). The sweep list is always length 5, ordered by `shift` ascending.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `version` | `payload["version"]` | — | Benchmark version |
| `local_align_canonical` | `mean_max_attn_to_target` at `shift = -1` | ↑ | Headline: how strongly the model attends to the immediate predecessor when that is the true structure |
| `local_align_shift_m2` | `mean_max_attn_to_target` at `shift = -2` | ↑ | Alignment to token two back |
| `local_align_shift_0` | `mean_max_attn_to_target` at `shift = 0` | ↑ | Self-attention (degenerate local) |
| `local_align_shift_p1` | `mean_max_attn_to_target` at `shift = +1` | ↑ | Alignment to successor |
| `local_align_shift_p2` | `mean_max_attn_to_target` at `shift = +2` | ↑ | Alignment to token two forward |
| `local_align_robustness` | `local_align_canonical / D` where `D = max(local_align_shift_0, local_align_shift_p1, local_align_shift_p2)`; if `D ≤ eps` the metric is `0.0` | ↑ ∈ [0, ∞) | Ratio of canonical alignment to strongest off-diagonal distractor; >1 means the head prefers the correct local structure |
| `local_align_peak_canonical` | `frac_peak_on_target` at `shift = -1` | ↑ | Fraction of queries where predecessor is the *top* attended token |
| `local_align_entropy_canonical` | `mean_entropy` at `shift = -1` | ↓ | Sharpness of attention at canonical shift (lower = better) |
| `linear_baseline_canonical` | `1 / (T-1)` ≈ 0.032 | — | Uniform attention baseline for `mean_max_attn_to_target` |
| `lift_over_uniform_canonical` | `local_align_canonical - linear_baseline_canonical` | ↑ | Absolute improvement over uniform |
| `random_baseline_peak_canonical` | `1 / T` ≈ 0.031 | — | Chance level for `frac_peak_on_target` |

**Full per-slice family.** In addition to the headline rows above, `score()` emits the per-slice triple for *every* shift in the sweep, named `local_align_shift_<tag>`, `local_align_entropy_shift_<tag>`, and `local_align_peak_shift_<tag>` (tag ∈ {`m2`, `m1`, `0`, `p1`, `p2`}, mapping `mean_max_attn_to_target` / `mean_entropy` / `frac_peak_on_target` respectively). The canonical-shift values are *also* exposed under the `_canonical` aliases listed above. These extra slice metrics are additive and do not require a `VERSION` bump.

**Edge cases.** If any denominator is zero, the corresponding ratio metric is set to `0.0` and a `ValueError` is *not* raised — the metric dict will still contain the key with value `0.0`.

## Bump procedure

Bump `VERSION` in `benchmark.py` and update this README when:
- The sweep axis changes (add/remove shifts, change ordering).
- Any metric formula changes.
- A payload key is renamed, removed, or retyped.
- The canonical condition (sequence length, vocab size, measured head) changes.

Do **not** bump when adding a new metric that doesn't alter existing ones.