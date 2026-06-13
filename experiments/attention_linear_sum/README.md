# Goal: attention_linear_sum

## Question
Can an attention head faithfully compute a **linear combination** of two scalar features — `y = α·x₁ + β·x₂` — broadcast to every token position, when the coefficients (α, β) are supplied only in the query/key projections?

## Setup
Synthetic generator. No trained model. Each example is a batch of sequences where:
- Two scalar features `x₁, x₂` are embedded in the residual stream at positions 0 and 1.
- A "coefficient token" at position 2 carries (α, β) in its residual vector.
- The target at every position t ≥ 3 is `α·x₁ + β·x₂`.
- The attention head under test receives the whole sequence; its **output at positions t ≥ 3** is the measurement.

Canonical measurement condition (used for the headline metric):
- Sequence length 8, batch size 256.
- x₁, x₂ ~ Uniform[-1, 1]; α, β ~ Uniform[-2, 2].
- Single head, d_model=32, d_head=32, no MLP, no LayerNorm.
- The model function is **only the attention head** (Q/K/V/O projections + softmax + weighted sum).
- Canonical coefficients: **α = β = 1**.
- 24 coefficient pairs swept: α,β ∈ {0, ±1, ±2} × {0, ±1, ±2} excluding (0,0).

## Model Function Signature
```python
def model_fn(batch: Batch) -> np.ndarray:
    """
    Args:
        batch: Batch with fields
            x1: (B, 1)    # feature 1 at pos 0
            x2: (B, 1)    # feature 2 at pos 1
            alpha: (B, 1) # coefficient α at pos 2
            beta:  (B, 1) # coefficient β at pos 2
    Returns:
        out: (B, 5)  # model's scalar output at the 5 target positions t=3..7
    """
```
T = 8. The first three positions are context; positions 3–7 are targets. The attempt implements the attention head (or a full model reduced to head-equivalent computation) and returns predictions for the 5 target positions as a `(B, 5)` array. `task.evaluate` raises `ValueError` if the returned shape is not `(B, 5)`.

## Payload Contract
`task.evaluate` returns a dict with exactly these keys:
```python
{
    "version": int,                    # benchmark.VERSION
    "canonical": {
        "pred": (B, 5),                # predictions at Target positions 3-7 (canonical α,β)
        "target": (B, 5),              # α·x₁ + β·x₂
    },
    "sweep": [                         # 24 records, one per (α,β) pair
        {
            "alpha": float,            # α value
            "beta":  float,            # β value
            "mse": float,              # mean squared error over batch & 5 positions
            "mae": float,              # mean absolute error
            "r2":  float,              # R² score (1 - MSE/Var(target))
        },
        ...
    ],
    "config": {
        "seq_len": 8,
        "batch_size": 256,
        "d_model": 32,
        "d_head": 32,
        "num_target_positions": 5,
    },
    "baseline": {                      # mean-target reference predictor (canonical α,β)
        "mse_canonical": float,        # MSE of constant mean(target) predictor
        "r2_canonical":  float,        # R² of mean predictor (≈ 0 by construction)
    }
}
```
All arrays are **Python lists** (JSON-serialisable), not numpy arrays. `task.evaluate` handles conversion.

## Metrics (from benchmark.score)
| metric | formula | direction |
|--------|---------|-----------|
| `linear_combination_r2_canonical` | R² on canonical α,β (1,1) | bigger better |
| `linear_combination_mse_canonical` | MSE on canonical | smaller better |
| `linear_combination_r2_alpha_<a>_beta_<b>` | per-slice R²; floats as `1p0`, `m1p0`, `2p0`, `m2p0` | bigger better |
| `linear_combination_mse_alpha_<a>_beta_<b>` | per-slice MSE | smaller better |
| `linear_combination_robustness` | min(R²) / max(R²) across 24 slices; ≤ 1 (≈1 = uniform, can be negative if a slice does worse than its mean; 0 if max R² ≤ 0) | bigger better |
| `linear_baseline_r2_canonical` | R² of mean-target predictor | bigger better (reference) |
| `linear_baseline_mse_canonical` | MSE of mean-target predictor | smaller better (reference) |
| `lift_over_baseline_r2` | `r2_canonical - baseline_r2_canonical` | bigger better |

`version` is always the first key in the returned dict.

## Bump Procedure
- VERSION=1 initially.
- Bump if: any metric formula changes, payload keys change, canonical (α,β) changes, sweep grid changes.
- Do not bump for adding new per-slice metrics or optional payload fields.

## Example Interpretation
- `linear_combination_r2_canonical ≈ 1.0` → head perfectly computes x₁+x₂.
- `linear_combination_robustness ≈ 1.0` → performance uniform across all coefficients.
- `lift_over_baseline_r2 > 0` → mechanism beats trivial mean prediction.