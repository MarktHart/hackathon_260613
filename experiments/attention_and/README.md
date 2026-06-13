# attention_and

## Question

Can a transformer attention head implement a logical **AND** over two features that are stored in **superposition** in the residual stream? Concretely: when two query-direction features `A` and `B` are both present (each at cosine similarity ≥ threshold with their respective query vectors), does the attention head output a sharp "both present" signal, while suppressing the signal when only one or neither is present?

This is a minimal test of **non-linear feature binding** in attention — a prerequisite for compositional reasoning.

---

## Setup

**Synthetic generator.**  
We construct a controlled residual-stream vector `x ∈ ℝ^d` as a linear combination of two orthogonal feature directions `v_A, v_B ∈ ℝ^d` plus isotropic noise:

```
x = α·v_A + β·v_B + ε·η,   η ~ N(0, I_d)
```

- `d = 128` (fixed).
- `v_A, v_B` are fixed random unit vectors with `v_A ⋅ v_B = 0`.
- `α, β ∈ {0, 1}` control presence/absence of each feature.
- `ε = 0.1` (fixed noise scale).
- The **canonical measurement condition** uses `α = β = 1` (both features present) and sweeps the cosine similarity between the *query* vectors `q_A, q_B` and their target features `v_A, v_B`.

**Query vectors.**  
For each sweep value `c = cos(q_A, v_A) = cos(q_B, v_B) ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}`, we construct query vectors `q_A, q_B` that have cosine `c` with their target features and are orthogonal to the other feature and to each other. The attention head computes:

```
attn_A = softmax((x ⋅ q_A) / √d)   # scalar in this 1-head, 1-query setup
attn_B = softmax((x ⋅ q_B) / √d)
```

The **model function** provided by an attempt must implement the *entire* attention computation (QKV projections, softmax, output projection) for a single head. The goal infrastructure handles the residual-stream construction and the sweep.

**Model function signature (contract with attempts):**

```python
def model_fn(q_A: np.ndarray, q_B: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Compute attention weights for the two query vectors on a single residual stream vector.

    Args:
        q_A: (d,) query vector for feature A.
        q_B: (d,) query vector for feature B.
        x:   (d,) residual stream vector (α·v_A + β·v_B + noise).

    Returns:
        (attn_A, attn_B) where each is the softmax attention weight (scalar in [0,1])
        that the head assigns to the token position (here only one position, so
        effectively the pre-softmax logit passed through softmax over a 1-element
        sequence — i.e. σ(logit) = 1.0 always. The meaningful signal is the *logit*
        x⋅q; see payload contract below).
    """
    ...
```

*Clarification:* Because we use a single-token sequence, softmax is degenerate. The payload therefore records the **pre-softmax logits** `logit_A = x ⋅ q_A / √d` and `logit_B = x ⋅ q_B / √d`. The attempt's `model_fn` returns these logits (or any monotonic transform; the benchmark only uses their difference). The infrastructure calls `model_fn` for each `(c, α, β)` combination.

---

## Canonical Measurement Condition

| Parameter          | Value                     |
|--------------------|---------------------------|
| Residual dimension | `d = 128`                 |
| Feature directions | Fixed orthogonal `v_A, v_B` (seeded) |
| Noise scale        | `ε = 0.1`                 |
| Sweep axis         | `c = cos(q, v) ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}` |
| Presence pairs     | `(α, β) ∈ {(1,1), (1,0), (0,1), (0,0)}` |
| Seeds              | `generate(seed)` fixes `v_A, v_B, η` for each `(α,β)`; `seed=0` is canonical |

---

## Payload Contract

`task.evaluate(model_fn)` returns a dict with the following exact structure:

```python
{
    "version": 2,                   # payload schema version (matches benchmark.VERSION)
    "d": 128,                       # residual dimension
    "noise_scale": 0.1,             # ε
    "sweep": [                      # one record per cosine value c
        {
            "cos_sim": 0.0,         # c = cos(q_A, v_A) = cos(q_B, v_B)
            "logit_AA": float,      # x(1,1) ⋅ q_A / √d  (both features present)
            "logit_AB": float,      # x(1,1) ⋅ q_B / √d
            "logit_A0": float,      # x(1,0) ⋅ q_A / √d  (only A present)
            "logit_B0": float,      # x(0,1) ⋅ q_B / √d  (only B present)
            "logit_00_A": float,    # x(0,0) ⋅ q_A / √d  (neither present)
            "logit_00_B": float,    # x(0,0) ⋅ q_B / √d
        },
        ...                         # repeated for each c in the sweep
    ],
    "canonical_cos_sim": 0.7        # the c value used for headline metrics
}
```

**Notes:**

- All logits are **pre-softmax** scalars (`x ⋅ q / √d`). Each reported value is the **mean over independent noise draws** of the same residual configuration (the infrastructure averages internally for stability; the payload still carries one float per quantity).
- `logit_AA` and `logit_AB` are the two logits when *both* features are present (`α=β=1`).
- `logit_A0` is the logit for `q_A` when only `A` is present (`α=1, β=0`).
- `logit_B0` is the logit for `q_B` when only `B` is present (`α=0, β=1`).
- `logit_00_A`, `logit_00_B` are the noise-floor logits when neither feature is present.
- The sweep contains exactly 6 records (one per `c` value). Order is fixed: ascending `cos_sim`.

---

## Metrics

All metrics are **bigger-is-better** unless noted.

### Headline summary

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `superposition_robustness` | `mean_c( and_sharpness_c ) / and_sharpness_c=1.0` | Average AND sharpness across the sweep, normalised by the perfect-alignment (`c=1.0`) sharpness. Range `[0, 1]`. Measures how gracefully the AND degrades as query alignment worsens. |

### Per-slice values (one per `cos_sim` value `c`)

For each `c`, define the **AND sharpness** as the logit gap between the "both present" case and the maximum of the single-feature cases:

```
and_sharpness(c) = logit_AA(c) - max(logit_A0(c), logit_B0(c))
```

(Using the `logit_AB` symmetric counterpart gives the same value in expectation; we average the two heads for stability.)

| Metric key | Formula | Units |
|------------|---------|-------|
| `and_sharpness_cos_<c>` | `(logit_AA - max(logit_A0, logit_B0) + logit_AB - max(logit_A0, logit_B0)) / 2` | logits (`x⋅q/√d`) |
| `and_sharpness_cos_0p0` … `and_sharpness_cos_1p0` | as above for each `c ∈ {0.0, 0.3, 0.5, 0.7, 0.9, 1.0}` | logits |

*Float formatting in keys:* `0.0 → 0p0`, `0.3 → 0p3`, `0.7 → 0p7`, `1.0 → 1p0`.

### Reference baselines (linear no-mechanism strawman)

A linear baseline has no AND non-linearity: it is an **additive** probe that reads both features with weight `c` and sums them — no interaction term. Reported in the same `x·q/√d` units as the model logits, its expected logits are `(α·c + β·c)/√d` (noise has mean 0), giving:

```
linear_baseline_sharpness(c) = (2c − max(c, c)) / √d = c / √d
```

The benchmark computes this analytically from `payload["d"]`. A genuine AND mechanism must **beat** this additive floor (`lift_over_linear > 0`) by suppressing the single-feature responses below what mere additivity predicts.

| Metric key | Meaning |
|------------|---------|
| `linear_baseline_sharpness_cos_<c>` | Expected sharpness of a linear probe under identical conditions. |
| `lift_over_linear_cos_<c>` | `and_sharpness_cos_<c> - linear_baseline_sharpness_cos_<c>` (positive = mechanism beats linear). |

### Canonical single-number views (convenience, not additional information)

| Metric key | Source |
|------------|--------|
| `and_sharpness_canonical` | `and_sharpness_cos_0p7` |
| `linear_baseline_sharpness_canonical` | `linear_baseline_sharpness_cos_0p7` |
| `lift_over_linear_canonical` | `lift_over_linear_cos_0p7` |

---

## Bump Procedure

- **VERSION 1** (historical): Only measured at `c = 0.0` (orthogonal queries). Payload had no sweep, only a single record.
- **VERSION 2** (current): Added the `cos_sim` sweep, per-slice metrics, and `superposition_robustness`. Payload structure changed incompatibly.
- Future bumps required when: any metric formula changes, payload keys are renamed/removed/retyped, or the canonical `cos_sim` (currently `0.7`) changes.
- Not required when: new metrics are added without touching existing ones, or a new `cos_sim` value is appended to the sweep (the sweep is extensible by design).

---

## Directionality Cheat Sheet

| Metric | Better |
|--------|--------|
| `superposition_robustness` | **Higher** (closer to 1) |
| `and_sharpness_cos_*` | **Higher** |
| `lift_over_linear_cos_*` | **Higher** (positive = non-linear AND works) |
| `linear_baseline_sharpness_cos_*` | Reference only (not optimised) |