# attention_lis — Linear Independent Subspaces in Attention

## Question

Do attention heads naturally organize distinct semantic features into **linearly independent (orthogonal) subspaces** of the query/key space, and how robust is this separation under distributional shift?

## Setup

**Synthetic generator.** We construct sequences where each position carries a *feature vector* `f ∈ ℝ^d_feat` composed of `K` independent binary factors (e.g. colour, shape, size). The generator emits:

- `tokens`: integer IDs for a small vocabulary (factor combinations → token).
- `factors`: ground-truth factor matrix `(L, K)` with values `{-1, +1}`.
- `factor_directions`: `K` random orthogonal directions in `ℝ^d_model` that the *ideal* encoder would use to represent each factor.

The generator is deterministic given `seed`. The canonical configuration is:

| parameter | value |
|-----------|-------|
| `seq_len` | 128 |
| `d_model` | 64 |
| `d_feat`  | 64 |
| `K`       | 4 |
| `vocab_size` | 16 (2^K) |
| `noise_std`  | 0.1 |

**Canonical measurement condition.** Every attempt must evaluate on the **canonical batch** (`seed=0`) and on a **robustness sweep** over `noise_std ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 1.0}`. The sweep measures how subspace independence degrades as the input representation gets noisier.

## Model function signature

```python
def model_fn(tokens: np.ndarray,  # (L,) int32
             return_qk: bool = True) -> dict:
    """
    Returns:
        q: (L, d_model) float32  — queries for each position
        k: (L, d_model) float32  — keys for each position
        v: (L, d_model) float32  — values for each position (optional)
        attn: (L, L) float32     — attention weights (optional)
    """
```

Only `q` and `k` are required. The benchmark projects `q` and `k` onto the ground-truth `factor_directions` and measures linear independence.

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                          # mirrors benchmark.VERSION
    "config": {
        "seq_len": 128,
        "d_model": 64,
        "d_feat": 64,
        "K": 4,
        "vocab_size": 16,
        "canonical_noise_std": 0.1,
    },
    "canonical": {
        "q_proj": np.ndarray,   # (K, L) — q @ factor_directions.T
        "k_proj": np.ndarray,   # (K, L)
        "factor_directions": np.ndarray,  # (K, d_model)
        "factors": np.ndarray,  # (L, K) ground truth {-1,+1}
        "noise_std": 0.1,
    },
    "sweep": [
        {
            "noise_std": 0.0,
            "q_proj": np.ndarray,  # (K, L)
            "k_proj": np.ndarray,
        },
        {"noise_std": 0.1, ...},
        {"noise_std": 0.3, ...},
        {"noise_std": 0.5, ...},
        {"noise_std": 0.7, ...},
        {"noise_std": 1.0, ...},
    ],
    "factor_directions": np.ndarray,  # (K, d_model) — same for all
    "factors": np.ndarray,            # (L, K) — same for all (canonical factors)
}
```

All arrays are `float32` unless noted. `q_proj[k, i] = q[i] @ factor_directions[k]`.

## Metrics

`benchmark.score(payload)` returns a flat dict:

| metric | formula | bigger is better? |
|--------|---------|-------------------|
| `lis_orthogonality_canonical` | mean over `k≠k'` of `1 - |cos(W_k, W_k')|`, where `W_k ∈ ℝ^K` is the model's **encoding direction** for factor `k` — the difference of class means of the projected representation `q_proj` between positions with `factor_k=+1` and `factor_k=-1`. Orthogonal encoding directions ⇒ linearly independent subspaces. A degenerate (≈0-norm) `W_k` is treated as maximally aligned (contributes 0). | ✓ |
| `lis_orthogonality_noise_0p0` … `lis_orthogonality_noise_1p0` | same, per sweep entry | ✓ |
| `lis_robustness` | `min(orthogonality_noise_x) / orthogonality_noise_0p0` ∈ [0,1] | ✓ |
| `lis_alignment_canonical` | mean_k `corr(Q_k, factor_k)` — how well each query direction matches its ground-truth factor | ✓ |
| `lis_alignment_noise_0p0` … | per sweep entry | ✓ |
| `linear_baseline_orthogonality_canonical` | same encoding-direction orthogonality, but for a no-mechanism model: i.i.d. Gaussian `q` independent of the factors (fixed seed). Random `q` ⇒ random encoding directions ⇒ low orthogonality. | ✓ |
| `lift_over_linear_baseline_canonical` | `lis_orthogonality_canonical - linear_baseline_orthogonality_canonical` | ✓ |
| `version` | `benchmark.VERSION` | — |

**Interpretation.** `lis_orthogonality` near 1 means the `K` factor encoding directions are mutually orthogonal (perfect LIS); a structureless or zero model scores at/below the linear baseline. `lis_alignment` near 1 means each query direction correlates with exactly one ground-truth factor. `lis_robustness` near 1 means orthogonality survives noise. The linear baseline uses i.i.d. Gaussian `q` independent of the factors, scored under the identical encoding-direction metric — a real method is only meaningful if it beats it.

## Bump procedure

Bump `VERSION` in `benchmark.py` and update this README when:
- the payload keys, types, or canonical config change;
- any metric formula changes;
- the sweep axis or values change.

Adding a new metric (e.g. `lis_orthogonality_keys`) does **not** require a bump.