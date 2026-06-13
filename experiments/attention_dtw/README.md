# attention_dtw

## The question

When two sequences are **time-warped** versions of one another (the same
underlying signal sampled along a non-linear, monotone schedule), does a
model's attention learn to **align** them the way Dynamic Time Warping (DTW)
would? Concretely: for each *key* position, the head should attend most
strongly to the *query* position it is genuinely aligned to, and the induced
correspondence should be **monotone** (no crossing alignments) — exactly the
constraints DTW enforces.

A head that has discovered an alignment circuit tracks the ground-truth warp
path. A head that merely runs down the diagonal does fine when the two
sequences are not warped, but degrades as the warp grows. The headline metric
measures how much of the alignment quality is **retained under heavy warping**
— i.e. whether the mechanism is a real alignment circuit or a diagonal
shortcut.

## Setup — synthetic

Fully synthetic, deterministic generator (no trained model, no dataset). For
each example:

1. A query sequence `queries` of `M = 16` random feature vectors in `R^D`
   (`D = 8`) is drawn.
2. A monotone non-decreasing **alignment** `align: key_index -> query_index`
   of length `N = 20` is drawn. Its non-linearity is controlled by a scalar
   `warp >= 0`. At `warp = 0` the alignment is the straight diagonal; larger
   `warp` bends it.
3. The key sequence `keys` is built by copying the aligned query vectors and
   adding small Gaussian noise: `keys[n] = queries[align[n]] + eps`.

So the ground-truth alignment between `keys` and `queries` is known exactly,
and a correct head's argmax over query positions should equal `align[n]`.

We sweep `warp` over `(0.0, 0.25, 0.5, 0.75)`. The **canonical measurement
condition** is `warp = 0.5`.

## The model function — the contract with attempts

An attempt provides a single callable:

```python
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

def model_fn(queries, keys) -> attn:
    # queries: float64 array, shape (M, D) = (16, 8)
    # keys:    float64 array, shape (N, D) = (20, 8)
    # returns: float array, shape (num_heads, N, M)
    #          attn[h, n, m] = attention weight from key n to query m
    #          each row attn[h, n, :] must sum to 1 (row-stochastic over M)
    ...
```

A 2-D return of shape `(N, M)` is accepted and treated as a single head.
`num_heads` is inferred from the first call and must be constant across calls.
`evaluate` keeps, per example, the **best head** (the head whose argmax path
best matches the ground truth) — the interpretability framing is "does *some*
head implement alignment", so a method is not penalised for having other heads
do other things.

`task.py` also exports `random_model_fn()` returning a random row-stochastic
`ModelFn` (pure NumPy) used only by the pipeline smoke test.

## Payload contract

`task.evaluate(model_fn)` returns exactly:

```python
{
  "version":         1,                 # int, == benchmark.VERSION
  "setup":           "synthetic_dtw_alignment",   # str label
  "num_heads":       int,               # heads inferred from model_fn output
  "seq_len_q":       16,                # M
  "seq_len_k":       20,                # N
  "feature_dim":     8,                 # D
  "n_examples":      int,               # examples per warp slice
  "canonical_warp":  0.5,               # float, present in warp_sweep
  "warp_sweep":      [0.0, 0.25, 0.5, 0.75],      # list[float]

  "sweep": [          # one record per warp, same length/order as warp_sweep
    {
      "warp":             float,   # the warp value
      "best_head_overlap":float,   # mean over examples of best-head path overlap in [0,1]
      "mean_head_overlap":float,   # mean over examples & heads of path overlap in [0,1]
      "monotonicity":     float,   # mean over examples of best-head monotone-step fraction [0,1]
    }, ...
  ],

  "baseline": [       # one record per warp, same length/order as warp_sweep
    {
      "warp":             float,
      "diagonal_overlap": float,   # overlap of the fixed straight-diagonal alignment [0,1]
      "uniform_overlap":  float,   # chance overlap = 1 / M
    }, ...
  ],
}
```

`path_overlap` for one example = fraction of key positions `n` where
`argmax_m attn[h, n, m] == align[n]` (exact match). `monotonicity` = fraction
of consecutive key positions whose predicted query index is non-decreasing.

## Metrics (`benchmark.score`)

Flat dict of scalars. `version` is first. Bigger is better for every metric.

| metric | meaning |
|--------|---------|
| `path_overlap_canonical` | best-head overlap at `warp = 0.5` |
| `alignment_robustness` | **headline.** overlap at max warp / overlap at min warp, clipped to `[0,1]`. 1.0 = alignment fully retained under heavy warp |
| `path_overlap_warp_<w>` | per-slice best-head overlap |
| `mean_head_overlap_warp_<w>` | per-slice mean-over-heads overlap |
| `monotonicity_warp_<w>` | per-slice best-head monotonicity |
| `diagonal_baseline_overlap_warp_<w>` | per-slice overlap of the no-mechanism straight diagonal |
| `uniform_baseline_overlap_warp_<w>` | per-slice chance overlap (`1/M`) |
| `lift_over_diagonal_canonical` | `path_overlap_canonical` minus diagonal overlap at canonical warp |

Per-slice float values are encoded `0p25`-style (`warp_0p5`, `warp_0`).

## Pipeline hooks

- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken(metrics)` → `True` on any NaN/inf, or when the canonical
  overlap does not beat chance (`<= uniform_baseline_overlap` at canonical).

## Bump procedure

Bump `benchmark.VERSION` (and `payload["version"]`) when you change `M`, `N`,
`D`, the warp sweep, the canonical warp, the `model_fn` shape contract, or any
existing metric formula. Adding a new metric or an optional payload key does
not require a bump. Update this README's contract tables in the same commit.
