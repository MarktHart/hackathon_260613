# Attention Distance Comparison

## Question
How does a model's attention mass distribute as a function of token-to-token
positional distance? Does attention concentrate on *nearby* keys (a steep
distance decay), and do individual layers/heads differ in their distance
preference (local vs. global heads)?

## Setup
**Synthetic generator.** There is no trained model. `task.generate` produces a
deterministic batch of random token sequences; the attempt provides a
`model_fn` that returns per-query-normalised attention patterns for those
sequences. The benchmark bins every query→key weight by distance `|i - j|` and
measures how steeply the mean per-bin weight decays with distance, comparing
against a uniform-attention baseline.

## Canonical Measurement Condition
- Sequence length: **64**
- Batch size: **32** sequences
- Vocabulary size: **1000** (tokens are integers `0..999`)
- Number of layers: **4**
- Heads per layer: **8**
- Distance bin edges (half-open): `[0, 1, 2, 3, 4, 5, 7, 11, 17, 33, 64]`
  → 10 bins: `[0,1) [1,2) [2,3) [3,4) [4,5) [5,7) [7,11) [11,17) [17,33) [33,64)`
- Bin centers (x-axis for the decay fit):
  `[0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 9.0, 13.5, 24.5, 48.0]`
- Seed: **0** (fixed; `generate` ignores its `seed` argument but accepts it for
  API compatibility, so two attempts always score on identical data).

## Model Function Signature
```python
def model_fn(input_ids: np.ndarray) -> dict:
    """
    Args:
        input_ids: int array of shape (batch, seq_len) = (32, 64)
    Returns:
        {"attention": np.ndarray} where attention has shape either
            (n_layers, n_heads, batch, seq_len, seq_len)   # batched, or
            (n_layers, n_heads, seq_len, seq_len)          # broadcast over batch
        Each row over the final (key) axis must sum to 1 (atol 1e-3).
    """
    ...
```
The function must be pure and deterministic for a given input.

## Payload Contract
`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                                       # payload schema version (int)
    "canonical_config": {                               # self-describing; score() ignores it
        "seq_len": 64, "batch_size": 32,
        "n_layers": 4, "n_heads": 8, "vocab_size": 1000,
        "seed": 0,
        "distance_bin_edges": [0, 1, 2, 3, 4, 5, 7, 11, 17, 33, 64],
    },
    "distance_bins": list[float],                       # length 10, bin centers
    "mean_attn_per_bin": list[float],                   # length 10, global mean attn weight per bin
    "uniform_baseline_per_bin": list[float],            # length 10, == 1/seq_len in every bin
    "mean_attn_per_layer_head_bin": list[list[list[float]]],  # shape (4, 8, 10)
}
```

Semantics:
- `mean_attn_per_bin[b]`: mean attention weight of the cells whose distance
  falls in bin `b`, averaged over batch, query position, and all layers/heads.
- `uniform_baseline_per_bin[b]`: the same quantity under uniform attention
  (`1/seq_len` for every cell), a flat reference curve.
- `mean_attn_per_layer_head_bin[l][h][b]`: the per-bin mean for layer `l`,
  head `h` (averaged over batch and query position only).

## Metrics
`benchmark.score(payload)` returns a flat dict (all **bigger-is-better**):

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `version` | `payload["version"]` | Schema version (first key). |
| `distance_decay_slope_canonical` | negated OLS slope of `ln(mean_attn)` vs `log2(distance)` over bins with center ≥ 1 | **Headline.** Larger = attention falls off faster with distance (more local). |
| `uniform_baseline_decay_slope` | same fit on `uniform_baseline_per_bin` | Reference; ≈ 0 for a flat curve. |
| `lift_over_uniform_decay_slope` | `distance_decay_slope_canonical − uniform_baseline_decay_slope` | Decay attributable to the model, not the binning. |
| `local_attention_fraction_canonical` | `sum(mean_attn over bins with center ≤ 4.5) / sum(mean_attn)` | Fraction of per-bin mean mass on token distances ≤ 4 (the five singleton-distance bins 0–4). In `[0, 1]`. |
| `attn_entropy_canonical` | Shannon entropy (bits) of `mean_attn_per_bin` as a distribution | Reported as-is (not headline); lower = more peaked. |
| `mean_attn_dist_{d}` | `mean_attn_per_bin[b]` | Per-slice mean attention at each bin center `d` (int-formatted). |
| `layer_head_decay_slope_layer{L}_head{H}` | decay slope of `mean_attn_per_layer_head_bin[L][H]` | Per-head decay; lets the grader see which heads are local vs. global. |

The decay fit skips the distance-0 bin (center `0.5`), where the
self/adjacent cell dominates trivially.

## Bump Procedure
Bump `VERSION` in `benchmark.py` **and** `payload["version"]` in `task.py`
together when:
- bin edges or canonical `seq_len` / `batch_size` / `n_layers` / `n_heads` change;
- any payload key is added/removed/renamed/retyped;
- any existing metric formula changes.
Update this README's tables in the same commit. Old `benchmark.json` files
stay on disk; the dashboard filters to the highest `version`.

## Optional Pipeline Hooks
- `GPU_REQUIREMENT = 0` (pure NumPy, no model in the loop).
- `is_obviously_broken(metrics)` returns `True` when any metric is NaN/inf,
  when `local_attention_fraction_canonical` is outside `[0, 1]`, or when the
  headline decay slope does not exceed the uniform baseline (no distance
  mechanism beyond the binning artifact).
