# Goal: attention_dyck

## Question
Do attention heads in a transformer track the **stack depth** of a Dyck language (balanced parentheses) — i.e., does some head attend to the matching opening parenthesis for each closing parenthesis, or otherwise encode the current nesting level?

## Setup
- **Synthetic generator**: `task.generate(seed)` produces a `Batch` of Dyck-1 strings (single bracket type `(` `)`) with maximum depth ≤ 6, length ≤ 64.
- **Vocabulary**: `0 = PAD`, `1 = (`, `2 = )`, `3 = BOS`, `4 = EOS`.
- **Canonical measurement condition**: Evaluate on the held-out test split (seed=42) of 512 sequences. The model function receives `(input_ids, attention_mask)` and returns a dict with key `attn_weights` of shape `(batch, n_heads, seq_len, seq_len)` — the **post-softmax attention weights** from the final layer only. (If the model has multiple layers, the attempt chooses which layer to expose; document the choice in the attempt README.)

## Canonical model function signature
```python
def model_fn(input_ids: np.ndarray, attention_mask: np.ndarray) -> dict:
    # input_ids: (batch, seq_len) int32
    # attention_mask: (batch, seq_len) bool
    # returns {"attn_weights": (batch, n_heads, seq_len, seq_len) float32}
```

## Payload contract (output of `task.evaluate`)
```python
{
    "version": 1,
    "canonical_seed": 42,
    "seq_len": int,
    "max_depth": int,
    "n_heads": int,
    "n_layers": int,          # 1 if single layer exposed; attempt documents which layer
    "per_head": list[{
        "head": int,          # 0 .. n_heads-1
        "matching_accuracy": float,   # fraction of closing brackets where max attn is on matching open
        "depth_corr": float,          # mean Pearson r (over closings) between a closing's attention to each
                                      # open position and that open's nesting depth (open_depth, 1-indexed)
        "diag_frac": float,           # fraction of attn mass on diagonal (current token)
    }],
    "aggregated": {
        "best_matching_accuracy": float,
        "best_depth_corr": float,
        "linear_baseline_matching": float,   # matching accuracy of a fixed head attending uniformly to all
                                             # prior open brackets, computed on the canonical batch
    }
}
```
All floats are Python `float`. `per_head` length == `n_heads`.

## Metrics (output of `benchmark.score`)
| metric | formula | direction |
|--------|---------|-----------|
| `dyck_matching_canonical` | `aggregated.best_matching_accuracy` at canonical seed | bigger better |
| `dyck_depth_corr_canonical` | `aggregated.best_depth_corr` at canonical seed | bigger better |
| `dyck_head_<h>_matching` | `per_head[h].matching_accuracy` | bigger better |
| `dyck_head_<h>_depth_corr` | `per_head[h].depth_corr` | bigger better |
| `dyck_diag_frac_mean` | mean of `per_head[*].diag_frac` | smaller better (diagnostic) |
| `linear_baseline_matching` | `aggregated.linear_baseline_matching` — matching accuracy of a **fixed** head attending uniformly to all prior open brackets, computed deterministically on the canonical batch | bigger better (reference) |
| `lift_over_baseline_matching` | `dyck_matching_canonical - linear_baseline_matching` | bigger better |

## Bump procedure
- `VERSION` increments when: payload keys change type/name, canonical seed/max_depth/seq_len change, or metric formulas change.
- Adding a new per-head metric does **not** require a version bump.

## Random model smoke test
`task.random_model_fn()` returns uniform attention over valid positions (masked by `attention_mask`). The smoke test verifies `evaluate(random_model_fn())` produces a valid payload and `benchmark.score` returns all metrics without error.