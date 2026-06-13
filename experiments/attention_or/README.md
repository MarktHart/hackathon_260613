# attention_or

## Question

Does the model's attention mechanism implement a clean **logical OR** over two independent query directions? Concretely: when the query is a superposition of two orthogonal "feature" directions (q_A and q_B), does the attention score equal the **maximum** of the two individual scores (OR semantics), rather than their sum (linear superposition) or some other combination?

This mirrors the `attention_and` goal, but for the OR truth table:
- 0 OR 0 = 0
- 0 OR 1 = 1
- 1 OR 0 = 1
- 1 OR 1 = 1

A mechanism implementing OR should show **max-pooling** behaviour: the combined-query attention equals `max(score(q_A), score(q_B))` at every key position.

## Setup

**Synthetic generator only** — no trained model, no external weights.  
The generator constructs a tiny "toy" attention block, rebuilt at each sweep point:

- Two unit **query directions** `q_A, q_B ∈ ℝ^d` (d = 32) with a controlled
  cosine similarity `cos(q_A, q_B)` — this is the sweep parameter.
- Key matrix `K ∈ ℝ^{d×n}` (n = 64) whose first two columns are the signal keys:
  - `k_A` — aligned with `q_A` (column 0)
  - `k_B` — aligned with `q_B` (column 1)
  - Remaining `n-2` columns: random unit **noise** keys.
- The **balanced superposition query** `q_AB = (q_A + q_B) / ‖q_A + q_B‖` — the
  single combined query that an OR mechanism must make attend to *both* signal
  keys at once.

The sweep varies `cos(q_A, q_B) ∈ [0, 1]` across 11 evenly spaced points.  
Canonical condition: `cos(q_A, q_B) = 0.0` (orthogonal queries) — the clean
anchor where `q_AB` carries an equal, separable component of each direction.

The model function supplied by an attempt receives `(q, K)` and returns **pre-softmax attention scores** `s ∈ ℝ^n` (logits). The evaluator computes the three score vectors
`s_A = model_fn(q_A, K)`, `s_B = model_fn(q_B, K)`, `s_AB = model_fn(q_AB, K)`
and records the per-position values at the two signal keys (`k_A`, `k_B`) plus the max over all noise keys.

## Canonical measurement condition

- Query orthogonality: `cos(q_A, q_B) = 0.0`
- Key dimension: `d = 32`
- Sequence length: `n = 64`
- Seed: `0` (deterministic construction)
- Metric reported at this condition is the **headline** `or_sharpness_canonical`.

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                              # payload schema version
    "canonical_cos": 0.0,                      # the canonical sweep value
    "sweep": [                                 # one record per cos value
        {
            "cos": float,                      # cos(q_A, q_B) in [0, 1]
            "s_A_at_A": float,                 # s_A[0]  (score of q_A on k_A)
            "s_A_at_B": float,                 # s_A[1]  (score of q_A on k_B)
            "s_A_noise_max": float,            # max_{i≥2} s_A[i]
            "s_B_at_A": float,                 # s_B[0]
            "s_B_at_B": float,                 # s_B[1]
            "s_B_noise_max": float,            # max_{i≥2} s_B[i]
            "s_AB_at_A": float,                # s_AB[0]
            "s_AB_at_B": float,                # s_AB[1]
            "s_AB_noise_max": float,           # max_{i≥2} s_AB[i]
        },
        ...
    ],
    "model_config": {                          # metadata, not used by score()
        "d": 32,
        "n": 64,
        "seed": 0,
    }
}
```

All scores are **raw logits** (pre-softmax). The sweep contains 11 values:
`cos ∈ [0.0, 0.1, 0.2, ..., 1.0]`.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float]` with:

| Metric | Formula | Direction | Meaning |
|---|---|---|---|
| `version` | `payload["version"]` | — | Schema version for dashboard filtering |
| `or_sharpness_canonical` | `min(s_AB_at_A, s_AB_at_B) / max(s_A_at_A, s_B_at_B)` at `cos=0.0` | **Bigger is better** | Headline: how close the combined query gets to the ideal OR (max) at the orthogonal anchor. 1.0 = perfect max-pooling. |
| `or_sharpness_cos_<val>` | Same ratio at each sweep value (`val` formatted `0p0`, `0p1`, … `1p0`) | **Bigger is better** | Per-slice sharpness across the sweep. |
| `or_noise_leakage_cos_<val>` | `s_AB_noise_max / max(s_AB_at_A, s_AB_at_B)` | **Smaller is better** | Fraction of combined-query mass leaking to noise keys. |
| `linear_baseline_sharpness_cos_<val>` | Sharpness of the **linear superposition** baseline `s_lin = s_A + s_B` (same denominator) | **Bigger is better** | Reference: what a purely linear mechanism would score. |
| `lift_over_linear_canonical` | `or_sharpness_canonical − linear_baseline_sharpness_cos_0p0` | **Bigger is better** | Sharpness relative to the linear-superposition reference at the canonical condition (closer to 0 / positive = the combined query matches or beats running both queries separately). |
| `superposition_robustness` | `min_{cos} or_sharpness_cos_<val> / or_sharpness_canonical` | **Bigger is better** | Worst-case sharpness relative to canonical; 1.0 = flat curve. |

Edge cases:
- Denominators ≤ 0 → metric = 0.0 (not `inf`/`NaN`).
- Missing required keys, wrong `version`, or an empty `sweep` → `score()` raises
  `ValueError`/`KeyError` with a descriptive message (a hard failure is
  preferred over silently scoring garbage).

## Bump procedure

- `VERSION` in `benchmark.py` **must** be incremented when:
  - Any metric formula changes.
  - Payload keys are added/removed/renamed/retyped.
  - The canonical condition (`canonical_cos`, `d`, `n`, seed) changes.
- `README.md` benchmark-contract section must be updated in the same commit.
- Old `benchmark.json` files remain on disk; dashboard hides them by default.

## Model function signature (contract with attempts)

```python
def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    query: shape (d,)          -- single query vector
    keys:  shape (d, n)        -- key matrix
    returns: shape (n,)        -- pre-softmax attention scores (logits)
    """
    ...
```

Attempts implement this callable. `task.random_model_fn()` returns a compliant function that emits `np.zeros(n)` for smoke-testing.