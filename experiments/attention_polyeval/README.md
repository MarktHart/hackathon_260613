# attention_polyeval

## Question

Can attention heads implement polynomial functions of their inputs (e.g., x², x³) through superposition and non-linear composition? This goal measures the fidelity with which a model's attention mechanism evaluates polynomial functions compared to a linear baseline.

## Setup

**Synthetic generator.** No trained model is used. We construct a controlled input distribution and probe whether the attention mechanism (Q, K, V projections + softmax + output projection) can compute target polynomials.

- Input dimension: `d_model = 64`
- Number of heads: `n_heads = 4`
- Head dimension: `d_head = 16`
- Sequence length: `seq_len = 128`
- Target polynomials: `x`, `x²`, `x³` (degree 1, 2, 3)
- Input distribution: `x ~ Uniform[-1, 1]` per token, per feature

The task generator creates a batch of random inputs and the corresponding target polynomial values. The model function receives the raw inputs and must return attention outputs that approximate the targets.

## Canonical Measurement Condition

- Seed: `42`
- Input scale: `1.0` (Uniform[-1, 1])
- Polynomial degrees: `[1, 2, 3]`
- Number of tokens: `128`
- Number of features: `64`

## Model Function Signature

```python
def model_fn(inputs: np.ndarray) -> np.ndarray:
    """
    Args:
        inputs: float32 array of shape [seq_len, d_model], the token embeddings.
    Returns:
        outputs: float32 array of shape [seq_len, d_model], the attention block output.
    """
```

The attempt's `model_fn` should implement a full attention block (QKV projections, attention, output projection). The benchmark evaluates how well the output matches target polynomials applied elementwise to the input.

## Payload Contract

`task.evaluate` returns a dict with exactly these keys:

```python
{
    "version": int,                    # payload schema version (currently 1)
    "config": {
        "seed": int,
        "input_scale": float,
        "degrees": list[int],
        "seq_len": int,
        "d_model": int,
        "n_heads": int,
        "d_head": int,
    },
    "sweep": list[{
        "degree": int,                 # polynomial degree (1, 2, or 3)
        "mse": float,                  # mean squared error vs target polynomial
        "correlation": float,          # Pearson correlation with target
        "variance_explained": float,   # R² = 1 - MSE / Var(target)
    }],
    "linear_baseline": list[{
        "degree": int,
        "mse": float,
        "correlation": float,
        "variance_explained": float,
    }],
}
```

All floats are Python `float` (not numpy scalars). The `sweep` and `linear_baseline` lists have the same length and corresponding degrees.

## Metrics

`benchmark.score` returns a flat dict:

| Metric | Formula | Direction |
|--------|---------|-----------|
| `version` | `payload["version"]` | — |
| `poly_mse_canonical` | MSE at degree=2 (the canonical non-linear case) | **smaller is better** |
| `poly_correlation_canonical` | Correlation at degree=2 | **bigger is better** |
| `poly_r2_canonical` | Variance explained at degree=2 | **bigger is better** |
| `linear_baseline_mse_canonical` | Linear baseline MSE at degree=2 | smaller is better |
| `linear_baseline_correlation_canonical` | Linear baseline correlation at degree=2 | bigger is better |
| `linear_baseline_r2_canonical` | Linear baseline variance explained at degree=2 | bigger is better |
| `nonlinear_lift_mse_canonical` | `linear_baseline_mse_canonical - poly_mse_canonical` | bigger is better |
| `nonlinear_lift_r2_canonical` | `poly_r2_canonical - linear_baseline_r2_canonical` | bigger is better |
| `poly_mse_degree_<d>` | MSE for each degree in sweep | smaller is better |
| `poly_correlation_degree_<d>` | Correlation for each degree | bigger is better |
| `poly_r2_degree_<d>` | Variance explained for each degree | bigger is better |
| `linear_baseline_mse_degree_<d>` | Linear baseline MSE per degree | smaller is better |
| `linear_baseline_correlation_degree_<d>` | Linear baseline correlation | bigger is better |
| `linear_baseline_r2_degree_<d>` | Linear baseline variance explained | bigger is better |
| `nonlinear_lift_mse_degree_<d>` | `linear_baseline_mse - poly_mse` (positive = improvement) | bigger is better |
| `nonlinear_lift_r2_degree_<d>` | `poly_r2 - linear_baseline_r2` (positive = improvement) | bigger is better |
| `poly_eval_headline` | `poly_r2_canonical - linear_baseline_r2_canonical` | **bigger is better** |

The headline metric `poly_eval_headline` measures how much better the model does than a linear baseline on the canonical degree-2 task. Positive values mean the attention mechanism captures non-linear structure.

The **linear baseline** is the best *affine* predictor `a·x + b` (slope and intercept fit globally to the flattened data) — the strongest a purely linear map can do. The intercept is essential: for even degrees `x` is linearly uncorrelated with `xᵈ`, so the best affine fit is the constant mean and `linear_baseline_r2 ≈ 0`. That is the correct "a linear map cannot capture this" reference, so any positive `poly_eval_headline` genuinely reflects non-linear structure.

## Bump Procedure

Bump `VERSION` in `benchmark.py` when:
- The payload keys change (add/remove/rename)
- The sweep structure changes (e.g., adding a new axis)
- Any metric formula changes

Update this README's payload contract and metrics tables in the same commit.