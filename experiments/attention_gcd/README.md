# attention_gcd

## Question
When a small transformer is shown an integer pair `(a, b)` as the sequence
`[a, b, SEP]`, does it represent or compute `gcd(a, b)` internally? Concretely:

1. **Attention** — does any head's weight *from* the `SEP` position *to* the two
   operand positions correlate with the true gcd?
2. **Residual stream** — is `gcd(a, b)` **linearly decodable** from the residual
   activation at the `SEP` position, beyond what a linear probe on the raw
   inputs `[a, b]` already achieves?

The headline asks the second, sharper question.

## Setup
- **Synthetic generator** (`task.generate`): batches of integer pairs with
  `1 ≤ a, b ≤ MAX_N`, plus their gcd labels. Deterministic per seed.
- **Model**: a small transformer (canonically 2 layers, 4 heads, `d_model=128`)
  that has consumed the sequence `[a, b, SEP]`. An attempt wraps its model in a
  `model_fn` (signature below). A trivial null model is provided as
  `task.random_model_fn` for the smoke test.
- **Canonical measurement condition**: `MAX_N = 100`, `batch_size = 512`,
  evaluation seed `42`. The batch is split 50/50 into train/test; all probes
  are fit on the train half and scored on the held-out test half. Every attempt
  must report the headline under this condition.

## Model function signature
```python
def model_fn(tokens: np.ndarray) -> dict:
    """
    tokens: int32 array [batch, seq_len=3], rows are [a, b, SEP].
    Returns a dict with:
      "attn_weights": list of length n_layers; each float array
                      [batch, n_heads, q_len, k_len]  (softmaxed attention)
      "resid_post":   list of length n_layers; each float array
                      [batch, seq_len, d_model]        (post-block residual)
    """
```
The `SEP` position is the last token (`sep_idx = seq_len - 1 = 2`). Operand
positions are `0` (a) and `1` (b). Architectures other than 2×4×128 are allowed;
just return correctly shaped lists.

## Payload contract (returned by `task.evaluate`)
```python
{
    "version": 1,
    "config": {
        "max_n": 100, "batch_size": 512,
        "n_layers": int, "n_heads": int, "d_model": int,
        "n_train": int, "n_test": int, "sep_idx": int,
    },
    "attn_corr": [[float, ...], ...],   # [n_layers][n_heads]: Pearson r between
                                        #   a head's mean SEP->operand weight and gcd
    "baseline_attn_corr": float,        # Pearson r between raw (a+b) and gcd
    "global": {
        "resid_r2":  [float, ...],      # per-layer test R² of gcd-from-residual probe
        "resid_acc": [float, ...],      # per-layer test accuracy (round(pred)==gcd)
        "baseline_r2":  float,          # test R² of gcd-from-raw-[a,b] probe
        "baseline_acc": float,          # test accuracy of that baseline probe
    },
    "sweep": [                          # one record per gcd bin (held-out test split)
        {
            "bin": "g1", "lo": 1, "hi": 1, "count": int,
            "resid_acc": [float, ...],  # per-layer test accuracy within this bin
            "baseline_acc": float,      # baseline probe accuracy within this bin
        },
        # bins: g1 (gcd==1), g2_3 (2..3), g4_8 (4..8), g9p (>=9)
    ],
}
```
Notes:
- Probes are ridge regressions (`λ=1`) on standardised features with a bias term.
- Per-slice accuracy is used (not R²) so zero-variance bins like `gcd==1` are
  always well defined. Accuracy is bigger-is-better.

## Metrics (returned by `benchmark.score`)
| metric | meaning | direction |
|--------|---------|-----------|
| `gcd_decodability` | **headline** = `gcd_resid_r2_canonical` | bigger better |
| `gcd_resid_r2_canonical` | max over layers of residual-probe test R² (clipped 0..1) | bigger better |
| `gcd_decode_acc_canonical` | max over layers of residual-probe test accuracy | bigger better |
| `gcd_attn_corr_canonical` | max over heads of `|attn_corr|` | bigger better |
| `linear_baseline_resid_r2_canonical` | raw-`[a,b]` probe test R² | reference |
| `linear_baseline_acc_canonical` | raw-`[a,b]` probe test accuracy | reference |
| `linear_baseline_attn_corr_canonical` | `|corr(a+b, gcd)|` | reference |
| `resid_r2_lift_over_baseline` | method R² − baseline R² | bigger better |
| `decode_acc_lift_over_baseline` | method acc − baseline acc | bigger better |
| `attn_corr_lift_over_baseline` | attn corr − baseline corr | bigger better |
| `gcd_decode_acc_<bin>` | per-slice best-layer accuracy (`g1`, `g2_3`, …) | bigger better |
| `gcd_decode_acc_baseline_<bin>` | per-slice baseline accuracy | reference |
| `gcd_decode_robustness` | min/max of per-bin accuracy over populated bins | bigger better (≈1 = stable) |

All metrics are bigger-is-better (reference baselines are for contrast). The
direction is consistent across the whole file.

## Edge cases
- Empty / undersized batches → probes return R²/acc of `0.0` rather than `inf`.
- Empty bins (`count == 0`) contribute `0.0` and are excluded from robustness.
- Zero `ss_tot` (constant targets) → R² defined as `0.0`.
- Constant attention (e.g. uniform) → correlation defined as `0.0`.

## Pipeline hooks
- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken`: flags NaN/inf, or a residual probe that fails to beat
  the raw-input baseline R² (no mechanism to interpret). Never trips on a
  borderline-but-real result.

## Bump procedure
Bump `VERSION` (currently `1`) on any payload key change, metric-formula change,
or change to the canonical condition (`MAX_N`, batch size, seed, train/test
split, or bin edges). Update this README's tables in the same commit.
