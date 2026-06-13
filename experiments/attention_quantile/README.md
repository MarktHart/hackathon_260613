# attention_quantile

## Question

Does the model's attention distribution over tokens follow a heavy-tailed (power-law-like) or light-tailed (exponential-like) distribution? Quantified by the **quantile ratio** — the ratio of the 90th-percentile attention weight to the 50th-percentile attention weight — across a sweep of temperature conditions spanning heavy-tail (sharpening) and light-tail (diffusing) regimes.

A high quantile ratio indicates a few tokens dominate attention (heavy tail); a low ratio indicates mass is spread more evenly (light tail). Mechanistic interpretability methods that claim to identify "important" tokens should correlate with the heavy-tail structure.

---

## Setup

**Synthetic generator.** We construct a fixed set of query/key vector pairs and assign each condition a temperature `scale` chosen so that a softmax over the dot-product logits emulates a heavy-tail (Pareto-like, sharpening) or light-tail (Exponential-like, diffusing) regime. The `scale` is the **only** per-condition signal `model_fn` receives — the `alpha`/`rate` values are condition labels, not inputs to the model. Scoring measures the concentration (quantile ratio) of whatever attention the attempt's `model_fn` produces; the benchmark does not compare against a fixed oracle. The generator produces:

- `queries`: `[n_queries, d_model]` — fixed random unit vectors
- `keys`: `[n_keys, d_model]` — fixed random unit vectors
- `logit_scales`: `[n_conditions]` — temperature scaling per condition
- `tail_type`: `"pareto"` | `"exponential"` — ground-truth tail family per condition

The canonical measurement condition uses `n_queries=32`, `n_keys=128`, `d_model=64`, and a sweep over `alpha ∈ {0.1, 0.3, 0.5, 0.7, 1.0}` (Pareto shape parameter) for heavy-tail and `rate ∈ {0.5, 1.0, 2.0, 5.0}` for light-tail. Each condition is a separate record in the sweep.

**Canonical measurement condition** (every attempt must use this):
- Model: the attempt provides a `model_fn` that takes `(queries, keys, scale)` and returns attention weights `[n_queries, n_keys]` (softmax-normalised per query).
- Sweep: 9 conditions (5 Pareto alphas + 4 Exponential rates).
- Temperature per condition (the `scale` passed to `model_fn`): Pareto `alpha=[0.1, 0.3, 0.5, 0.7, 1.0]` → `scale=[2.8, 2.4, 2.0, 1.6, 1.0]`; Exponential `rate=[0.5, 1.0, 2.0, 5.0]` → `scale=[0.8, 0.5, 0.3, 0.15]`. Heavier tail → higher temperature → sharper softmax.
- Seed: `seed=42` for `generate()`.

---

## Model Function Signature

```python
def model_fn(queries: np.ndarray, keys: np.ndarray, scale: float) -> np.ndarray:
    """
    Args:
        queries: [n_queries, d_model] float32
        keys:    [n_keys, d_model] float32
        scale:   float, temperature scaling for logits
    Returns:
        attn:    [n_queries, n_keys] float32, rows sum to 1.0
    """
```

The attempt's `main.py` implements this function using their method (e.g., a trained model's attention, a sparse approximation, a heuristic). `task.evaluate` calls it for each sweep condition and assembles the payload.

---

## Payload Contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```json
{
  "version": 1,
  "config": {
    "n_queries": 32,
    "n_keys": 128,
    "d_model": 64,
    "seed": 42
  },
  "sweep": [
    {
      "condition_id": "pareto_0p1",
      "tail_type": "pareto",
      "alpha": 0.1,
      "rate": null,
      "quantile_50": 0.0082,
      "quantile_90": 0.0417,
      "quantile_ratio": 5.12
    },
    {
      "condition_id": "pareto_0p3",
      "tail_type": "pareto",
      "alpha": 0.3,
      "rate": null,
      "quantile_50": 0.0079,
      "quantile_90": 0.0321,
      "quantile_ratio": 4.06
    },
    ...
  ]
}
```

**Per-record fields:**
| key | type | meaning |
|-----|------|---------|
| `condition_id` | str | unique identifier, format `{tail_type}_{param}` with floats as `0p7` |
| `tail_type` | str | `"pareto"` or `"exponential"` |
| `alpha` | float \| null | Pareto shape parameter (null for exponential) |
| `rate` | float \| null | Exponential rate parameter (null for pareto) |
| `quantile_50` | float | median attention weight across all query-key pairs |
| `quantile_90` | float | 90th-percentile attention weight |
| `quantile_ratio` | float | `quantile_90 / quantile_50` (headline per-condition metric) |

All floats are Python `float` (not numpy scalars).

**Zero-denominator handling.** If `quantile_50 == 0` (a very sparse attention with ≥50% of weights exactly zero), the denominator is floored to the smallest positive attention weight so `quantile_ratio` stays finite (never `inf`) and grows monotonically with concentration. Rows always sum to 1, so a positive weight always exists. The metric is most sensitive to top-decile concentration; ultra-sparse attention is still scored without crashing.

---

## Metrics

`benchmark.score(payload)` returns a flat dict:

| metric | formula | bigger-is-better |
|--------|---------|------------------|
| `version` | payload version | — |
| `quantile_ratio_canonical` | `quantile_ratio` at `pareto_0p5` (α=0.5) | yes (heavy tail → more interpretable structure) |
| `quantile_ratio_pareto_0p1` | per-slice | yes |
| `quantile_ratio_pareto_0p3` | per-slice | yes |
| `quantile_ratio_pareto_0p5` | per-slice | yes |
| `quantile_ratio_pareto_0p7` | per-slice | yes |
| `quantile_ratio_pareto_1p0` | per-slice | yes |
| `quantile_ratio_exponential_0p5` | per-slice | no (light tail → less structure) |
| `quantile_ratio_exponential_1p0` | per-slice | no |
| `quantile_ratio_exponential_2p0` | per-slice | no |
| `quantile_ratio_exponential_5p0` | per-slice | no |
| `pareto_vs_exponential_lift` | `mean(pareto ratios) / mean(exponential ratios)` | yes |
| `linear_baseline_quantile_ratio_canonical` | baseline from uniform attention | no (lower is better for baseline) |
| `lift_over_linear_baseline` | `quantile_ratio_canonical / linear_baseline_quantile_ratio_canonical` | yes |

**Baseline:** Uniform attention over `n_keys` gives `quantile_50 = quantile_90 = 1/n_keys`, so `quantile_ratio = 1.0`. Any method producing `ratio > 1.0` has non-uniform structure.

---

## Bump Procedure

- `VERSION` in `benchmark.py` increments when:
  - The payload keys change (add/remove/rename).
  - The canonical condition changes (n_queries, n_keys, d_model, seed, sweep values).
  - A metric formula changes.
- Adding a new per-slice metric or a new sweep condition (with extensible naming) does **not** require a bump.
- Update this README's payload/metric tables in the same commit as the `VERSION` bump.