# attention_scc: Attention Superposition Capacity Curve

## Question
How many independent features can a single attention head simultaneously hold
and still selectively attend to one of them? This goal measures the
**superposition capacity curve (SCC)** — how a head's ability to land its
attention mass on a target key degrades as the number of superimposed keys `K`
grows relative to the head dimension `d`.

## Setup
**Synthetic generator.** Each problem instance is a single noisy-retrieval task:

- `K` key vectors are sampled as random unit vectors in `R^d`.
- One of them is the **target**; the query `Q` equals the target key plus
  isotropic Gaussian noise at a fixed SNR, then renormalised.
- The remaining `K - 1` keys are distractors.
- The model must place as much attention mass as possible on the target index.

The **superposition ratio** is `rho = K / d`. We sweep `rho` from undercomplete
(`0.25`) to 4× overcomplete (`4.0`). This isolates the *geometric* capacity of
the head to resolve superposed features, independent of how they were learned.

## Canonical Measurement Condition
- Head dimension `d = 64`.
- Query noise SNR = 10 dB (`noise_var = 1 / (d · 10^(SNR/10))` per dimension).
- Sweep `rho ∈ [0.25, 0.5, 1.0, 2.0, 4.0]` → `K = [16, 32, 64, 128, 256]`.
- `100` random instances per `rho` (deterministic per-instance seeding).
- Measurement = target attention mass `attn[target_idx]`, averaged over the
  100 instances at each `rho`.

`task.generate(seed)` is deterministic. `task.evaluate` always uses `seed = 0`
(the canonical condition); the `seed` argument to `generate` exists for
diagnostics only.

## Model Function Signature
Attempts provide a callable:

```python
model_fn(Q: np.ndarray, K: np.ndarray) -> np.ndarray
```

- `Q`: shape `(d,)` — the query vector.
- `K`: shape `(K, d)` — the `K` unit-norm key vectors (rows).
- Returns: attention weights of shape `(K,)` — a **probability distribution**
  over the keys (finite, non-negative, sums to 1 within `rtol/atol = 1e-4`).

This is exactly the output of `softmax(Q @ K.T / sqrt(d))`. An attempt may
replace the standard computation with a mechanistic method (sparse coding,
learned projection, circuit-derived routing, …) but must return a valid
distribution. `evaluate` raises `ValueError` on shape, finiteness, sign, or
normalisation violations.

## Payload Contract
`task.evaluate(model_fn)` returns exactly:

```python
{
    "version": 1,
    "d": 64,
    "snr_db": 10.0,
    "sweep": [
        {
            "rho": 0.25,                  # K / d
            "K": 16,                      # number of keys
            "target_attention_mean": 0.0, # mean attn[target] over 100 instances
            "target_attention_std": 0.0,  # std across instances
            "chance_level": 0.0625,       # 1 / K (uniform-attention baseline)
        },
        # ... one record per rho in [0.25, 0.5, 1.0, 2.0, 4.0]
    ],
}
```

All numbers are plain Python `int`/`float`. `score()` consumes `version` and
`sweep`; `d`/`snr_db` are self-describing context.

## Metrics
Computed by `benchmark.score(payload)`. All are bigger-is-better.

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `version` | `1` | Benchmark version (dashboard filters to the max). |
| `scc_auc` | Trapezoidal area of `target_attention_mean` vs `rho`, normalised by the rho range. **Headline.** | Average target attention across the curve. `1.0` = perfect at all ratios, `~0` = chance. |
| `scc_auc_canonical` | `target_attention_mean` at `rho = 1.0` | Capacity at the complete (`K = d`) ratio. |
| `scc_rho_0p25` … `scc_rho_4p0` | `target_attention_mean` at each `rho` | Per-slice target attention mass. |
| `linear_baseline_scc_auc` | `scc_auc` formula applied to `chance_level` (uniform attention) | No-mechanism reference. |
| `linear_baseline_rho_<val>` | `chance_level` at each `rho` | Per-slice baseline (`1/K`). |
| `lift_over_linear_auc` | `scc_auc − linear_baseline_scc_auc` | Improvement over uniform attention. |
| `capacity_rho_0p5` | Largest `rho` with `target_attention_mean ≥ 0.5` (linear interpolation between sweep points) | "50% capacity" ratio. |
| `capacity_rho_0p9` | Largest `rho` with `target_attention_mean ≥ 0.9` | "90% capacity" ratio. |

Edge cases handled in `score()`:
- Empty/malformed sweep → `ValueError`/`KeyError`.
- Capacity threshold never reached → `min(rho)`; always exceeded → `max(rho)`.
- Single-point or zero-width rho range → degenerate AUC handled explicitly.
- Non-finite payload values → `ValueError`.

`is_obviously_broken` returns `True` on any NaN/inf metric or when `scc_auc`
fails to beat `linear_baseline_scc_auc` (no lift over uniform attention).

## Bump Procedure
Bump `VERSION` when: the swept `rho` values change, `d`/`snr_db` change, a
metric formula changes, or any payload key is renamed/retyped/removed. Adding a
new metric (e.g. `capacity_rho_0p75`) or an optional payload key does **not**
require a bump. Old `benchmark.json` files stay on disk; the dashboard filters
to the highest `VERSION`. Update this contract in the same commit as the bump.
