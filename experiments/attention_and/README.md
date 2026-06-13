# attention_and

## Question

Does a single attention head implement a **logical AND** between two query
vectors?  That is, when the head's query projects onto two stored keys
`k_A` and `k_B`, does the attention weight on the *joint* key `k_AND`
rise sharply only when **both** query components are present?

This goal measures the **sharpness** of that AND response as a function of
the cosine similarity between the two query components, and summarises it
as **superposition robustness** — the ratio of the weakest AND response
across the sweep to the strongest.

## Setup

**Synthetic generator.**  No trained model, no dataset.  The task constructs
a minimal 1-head, 1-layer attention block with fixed `Q`, `K`, `V` matrices
that *should* implement an ideal AND if the mechanism is clean:

- `d_model = 64`, `d_head = 64` (no head splitting)
- Two orthogonal *feature* directions `f_A`, `f_B` ∈ ℝ^64, `‖f‖ = 1`,
  `⟨f_A, f_B⟩ = 0`.
- Keys: `k_A = f_A`, `k_B = f_B`, `k_AND = (f_A + f_B) / √2` (normalised sum).
- Values: `v_A = f_A`, `v_B = f_B`, `v_AND = f_AND` (same direction as key).
- Query for a sweep point `c ∈ [0, 1]`:
  `q(c) = normalize( √(1-c) · f_A + √c · f_B )`.
  When `c = 0` the query is pure `f_A`; when `c = 1` it is pure `f_B`;
  intermediate `c` interpolates the *cosine similarity* between the two
  query components: `cos(q_A, q_B) = 2√(c(1-c)) - 1 ∈ [-1, 1]`.
- The attention block is run with `softmax(Q Kᵀ / √d)` and the weight on
  `k_AND` is recorded.

The *ideal* AND mechanism would put near-zero weight on `k_AND` for pure
`f_A` or pure `f_B`, and peak weight at `c = 0.5` (`cos = 0`).

## Canonical measurement condition

- Sweep `cos(q_A, q_B)` over 11 evenly spaced values in `[-1, 1]`:
  `[-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
- For each cosine value, compute the attention weight on `k_AND`.
- The *canonical* headline metric is reported at `cos = 0.0` (the
  orthogonal anchor, `c = 0.5`).

## Payload contract

`task.evaluate` returns a `dict` with exactly these keys:

```python
{
    "version": 2,                              # matches benchmark.VERSION
    "model_name": "synthetic_attention_and",   # fixed string
    "sweep": [                                 # length 11, ordered by cosine
        {
            "cos_qA_qB": float,                # cosine similarity in [-1, 1]
            "and_weight": float,               # attention weight on k_AND ∈ [0, 1]
            "a_weight": float,                 # attention weight on k_A
            "b_weight": float,                 # attention weight on k_B
        },
        ...
    ],
    "canonical_cos": 0.0,                      # the cosine used for _canonical metrics
}
```

- All weights are Python `float` (already reduced from tensors).
- `sum(and_weight, a_weight, b_weight) ≈ 1.0` for each record (softmax).
- The sweep order is fixed; `benchmark.score` relies on it.

## Metrics

| metric | formula | direction | interpretation |
|--------|---------|-----------|----------------|
| `and_sharpness_canonical` | `and_weight` at `canonical_cos = 0.0` | **bigger is better** | Peak AND response when the two query components are orthogonal. |
| `and_sharpness_cos_<val>` | `and_weight` at each sweep cosine | **bigger is better** | Per-slice view; `<val>` uses `0p7` notation (e.g. `cos_0p0`, `cos_n0p8`). |
| `linear_baseline_sharpness_cos_<val>` | `(1 - │cos│) / 2` at each cosine | **reference** | Weight a *linear* (non-AND) superposition would give to the midpoint key. |
| `superposition_robustness` | `min(and_sharpness_cos_*) / max(and_sharpness_cos_*)` | **bigger is better** | Ratio of weakest to strongest AND response across the full sweep. 1.0 = perfectly flat (bad); →0 = sharp peak only at orthogonality (good).  *Note: inverted from v1 so higher = more robust.* |
| `lift_over_linear_baseline_cos_<val>` | `and_sharpness_cos_<val> - linear_baseline_sharpness_cos_<val>` | **bigger is better** | How much the mechanism exceeds a linear superposition at each slice. |
| `version` | payload["version"] | — | echoed for dashboard filtering. |

## Bump procedure

`VERSION` is incremented when:
- any metric formula changes;
- a payload key is added, removed, or retyped;
- the canonical cosine or sweep grid changes.

After bumping, update this README's **Payload contract** and **Metrics**
tables in the same commit. Old `benchmark.json` files remain on disk; the
dashboard filters to the highest version by default.