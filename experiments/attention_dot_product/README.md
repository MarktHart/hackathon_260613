# attention_dot_product

## Question

Does an attempt's `model_fn` actually implement **scaled dot-product
attention** — `softmax(Q Kᵀ / √d_head) · V` — and how well does that fidelity
hold up as the **sequence length** (and therefore the softmax competition
between keys) grows?

A correct mechanism reproduces the reference attention output almost exactly at
every sequence length. A degenerate one (uniform/mean pooling, a linear map,
random output) collapses toward the no-dot-product baseline.

## Setup

**Synthetic generator** — fully controlled, no trained models. `Q`, `K`, `V`
are i.i.d. `N(0, 1)` tensors of shape `(batch, n_heads, seq_len, d_head)`. The
ground-truth output is the reference attention

```
gt_out = softmax(Q Kᵀ / √d_head) · V
```

We sweep `seq_len` to probe robustness as more keys compete for the softmax.

### Canonical measurement condition

- `d_head = 16`
- `n_heads = 4`
- `batch_size = 8`
- `canonical_seq_len = 32`
- `seq_len_sweep = [8, 16, 32, 64, 128]`
- generator seed `0` (deterministic; same seed → same tensors)

The **canonical** condition is the sweep record at `seq_len = 32`.

## Model function signature

```python
def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Args:
        Q, K, V: (batch, n_heads, seq_len, d_head) float arrays.
    Returns:
        out: (batch, n_heads, seq_len, d_head) — the attention output,
             same shape as V.
    """
```

The attempt returns the attention output directly; `task.evaluate` compares it
against the reference `gt_out` and reduces to per-slice scalars. `model_fn` is
called once per sweep value. Returning the wrong shape raises a `ValueError`.

## Payload contract

`task.evaluate` returns a dict with exactly these keys:

```python
{
    "version": 1,                                   # matches benchmark.VERSION
    "model_name": "synthetic_scaled_dot_product_attention",
    "config": {
        "d_head": 16,
        "n_heads": 4,
        "batch_size": 8,
        "canonical_seq_len": 32,
        "seq_len_sweep": [8, 16, 32, 64, 128],
    },
    "sweep": [                                      # one record per seq_len
        {
            "seq_len": 8,                           # sweep axis
            "mse": 1.2e-7,                          # mean squared error vs gt_out
            "rel_error": 0.004,                     # ‖pred-gt‖ / ‖gt‖ (Frobenius)
            "cos_sim": 0.999,                       # mean per-token cosine vs gt
            "baseline_mse": 0.31,                   # uniform-attention MSE (strawman)
        },
        ...                                         # one per seq_len in the sweep
    ],
}
```

Units / direction:

- `mse`, `rel_error` — error vs ground truth; **smaller is better** (≥ 0).
- `cos_sim` — mean per-token cosine similarity to `gt_out`; **bigger is
  better**, in `[-1, 1]`.
- `baseline_mse` — MSE of the uniform-attention strawman (mean of `V` over the
  key axis) against `gt_out`; a fixed reference, not optimised.

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars:

| metric | formula | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` | — |
| `attention_fidelity` | `clip(1 − mean(mse)/mean(baseline_mse), 0, 1)` over the sweep | **bigger = better** (headline) |
| `attention_fidelity_canonical` | same, at `canonical_seq_len` only | bigger = better |
| `cos_sim_canonical` / `mse_canonical` / `rel_error_canonical` / `baseline_mse_canonical` | canonical-slice values | cos bigger / errors smaller |
| `cos_sim_seqlen_<L>` | per-`seq_len` cosine | bigger = better |
| `mse_seqlen_<L>` | per-`seq_len` MSE | smaller = better |
| `rel_error_seqlen_<L>` | per-`seq_len` relative error | smaller = better |
| `baseline_mse_seqlen_<L>` | per-`seq_len` baseline MSE | reference |
| `cos_sim_worst` | `min` cosine across the sweep | bigger = better |
| `cos_sim_mean` | mean cosine across the sweep | bigger = better |
| `attention_robustness` | `clip(cos_sim_worst, 0, 1)` | bigger = better |
| `lift_over_baseline_canonical` | `baseline_mse_canonical − mse_canonical` | bigger = better |

### Headline summary

**`attention_fidelity`** — the fraction of the uniform-attention baseline error
removed by the attempt, averaged across the sequence-length sweep and clipped to
`[0, 1]`. `1.0` = perfect reconstruction of scaled dot-product attention; `0.0`
= no better than (or worse than) mean pooling.

### Edge cases

- Empty `sweep` → `score` raises `ValueError` (caught by validation).
- `baseline_mse == 0` (degenerate data) → the corresponding fidelity is `0.0`
  rather than a division by zero.
- `is_obviously_broken` short-circuits the jury when any metric is NaN/inf,
  when `attention_fidelity <= 0`, or when `cos_sim_worst < 0.1`.

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README, same commit) when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- the canonical condition (`canonical_seq_len`) or the sweep schema changes.

Do **not** bump when adding a new metric that leaves existing ones unchanged, or
when extending `seq_len_sweep` with another value (the sweep is extensible).
