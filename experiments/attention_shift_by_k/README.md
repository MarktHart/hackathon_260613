# Attention Shift by k

## Question
Does *some* attention head implement a clean "shift by k" operation — query
position `i` attending to key position `i - k`? We sweep the offset `k` and, for
each, report the single best head, so the metric asks whether a shift-by-k
mechanism *exists*, not whether the average head happens to do it.

## Setup
**Synthetic generator.** The shift-by-k pattern is purely positional, so the
ground-truth target is known exactly without any trained model.

- Vocabulary: integer token IDs in `[0, V)` with `V = 64`.
- Sequence length `L = 32`, batch of `B = 8` sequences.
- Tokens are drawn i.i.d. uniformly (seeded). Token identity does not affect the
  positional target — real models consume tokens, so genuine IDs are handed over.
- For offset `k`, the "correct" attention from query position `i` (with
  `k ≤ i < L`) places mass on key position `i - k`. Queries `i < k` have no valid
  target and are excluded from that slice's metric.

No trained model is required; the task is fully synthetic.

## Canonical measurement condition
- `L = 32`, `V = 64`, `B = 8`, seed `0`.
- Sweep over `k ∈ {1, 2, 3, 4, 8}` (5 conditions).
- **Canonical k = 1** (the previous-token shift) is the headline condition.
- `model_fn` receives token IDs `(B, L)` and returns attention `(B, H, L, L)`
  for any `H ≥ 1`. The evaluator renormalises each query row over keys, then for
  each `k` selects the head with the highest mass on the target.
- The same `B` sequences are used for every `k`; only the target offset changes.

## Payload contract
`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": int,            # == benchmark.VERSION (1)
    "model_name": str,         # free-form identifier from the attempt
    "seq_len": int,            # 32
    "batch_size": int,         # 8
    "vocab_size": int,         # 64
    "num_heads": int,          # H reported by the model
    "k_values": list[int],     # [1, 2, 3, 4, 8]
    "canonical_k": int,        # 1
    "uniform_baseline": float, # 1 / seq_len  (chance mass on one key)
    "sweep": [
        {
            "k": int,                     # shift offset
            "best_head_index": int,       # head with the highest target mass
            "best_head_mass": float,      # mean attn on key i-k for that head,
                                          #   over valid queries and batch
            "best_head_argmax_acc": float,# fraction of valid queries whose peak
                                          #   key == i-k, for that head
            "mean_head_mass": float,      # target mass averaged across all heads
            "uniform_baseline": float,    # 1 / seq_len
        }
        for k in (1, 2, 3, 4, 8)
    ],
}
```

All floats are plain Python `float`. `benchmark.score` validates that the sweep
is non-empty, that `canonical_k` appears in it, and that `k` values are unique.

## Metrics
`benchmark.score` returns a flat dict (all `*_mass` / `*_acc` / `*_lift` metrics
are **bigger-is-better**):

| Metric | Formula | Notes |
|--------|---------|-------|
| `version` | `benchmark.VERSION` | First key, for dashboard filtering. |
| `shift_robustness` | mean over `k` of `shift_lift_k_<k>` | **Headline.** Mean chance-normalised lift in `[0, 1]`. |
| `shift_mass_canonical` | `best_head_mass` at `k = 1` | Headline best-head mass at the canonical offset. |
| `shift_argmax_acc_canonical` | `best_head_argmax_acc` at `k = 1` | Peak-hit rate at canonical offset. |
| `lift_over_baseline_canonical` | `best_head_mass − baseline` at `k = 1` | Same units as mass. |
| `shift_mass_k_<k>` | `best_head_mass` for offset `k` | Per-slice best-head mass. |
| `shift_argmax_acc_k_<k>` | `best_head_argmax_acc` for offset `k` | Per-slice peak-hit rate. |
| `mean_head_mass_k_<k>` | `mean_head_mass` for offset `k` | Across-head average (diluted reference). |
| `shift_lift_k_<k>` | `(mass − base) / (1 − base)`, clipped `[0,1]` | Per-slice chance-normalised lift. |
| `linear_baseline_mass_k_<k>` | `1 / L` | Uniform (no-mechanism) reference, same conditions. |
| `uniform_baseline` | `1 / L` | The chance mass on a single key. |

Per-slice keys use the integer `k` directly (`shift_mass_k_1`, `shift_mass_k_8`,
…) since offsets are integers.

`is_obviously_broken` returns `True` on any NaN/inf metric, or when
`shift_mass_canonical ≤ uniform_baseline × 1.1` (the best head barely beats
chance at the canonical offset) — this skips the jury on degenerate runs only.

## Bump procedure
- Increment `VERSION` in `benchmark.py` if any metric formula changes, payload
  keys are added/removed/renamed, or the canonical condition (`k` sweep,
  `canonical_k`, `L`, `V`, `B`) changes.
- Update this README's "Payload contract" and "Metrics" tables in the same commit.
- Old `benchmark.json` files remain on disk; the dashboard filters to the highest
  `version` present.

## Model function signature
```python
def model_fn(input_ids: np.ndarray) -> np.ndarray:
    """
    Args:
        input_ids: int32 array of shape (B, L), token IDs in [0, vocab_size).

    Returns:
        attn: float array of shape (B, H, L, L) where
              attn[b, h, i, j] = attention weight from query i to key j
              for batch item b, head h. Rows need not sum to 1 over j
              (the evaluator renormalises per query position).
    """
```
Attempts implement this in their `main.py` and pass it to `task.evaluate`.
