# attention_or

## Question

Can a single attention head implement the logical **OR** of two binary features in superposition, and how does interference between the feature query directions degrade that computation?

We measure whether the attention output cleanly separates the `(0,0)` case (output ≈ 0) from the three `(1,*)`, `(*,1)` cases (output ≈ 1) as the cosine similarity between the two feature query vectors varies.

## Setup

**Synthetic generator.** No trained model is used. The task constructs a minimal 1-head attention block with:
- Two binary features `A`, `B ∈ {0,1}`.
- Query vectors `q_A`, `q_B` in `ℝ^d` (`d = 32` by default) with controllable cosine similarity `cos(q_A, q_B) = ρ`.
- Key/value projections that implement `OR(A, B)` in the absence of interference.
- A single attention head with no MLP, no layer norm, no residual stream — just `softmax(QK^T/√d)V`.

The generator produces a batch of all four input combinations `{(0,0), (0,1), (1,0), (1,1)}` with equal weight. The attention pattern is deterministic given `q_A`, `q_B`, `k_A`, `k_B`, `v_A`, `v_B`.

**Canonical measurement condition:**
- Dimension `d = 32`
- Key vectors `k_A = q_A`, `k_B = q_B` (matched queries/keys)
- Value vectors `v_A = v_B = [1, 0, ..., 0]` (scalar 1 in first component, zero elsewhere)
- Sweep axis: `ρ = cos(q_A, q_B) ∈ {0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95}`
- Canonical slice: `ρ = 0.7` (moderate interference, matching `attention_and`)

The attempt's `model_fn` receives the generated `Batch` (containing `q_A`, `q_B`, `k_A`, `k_B`, `v_A`, `v_B`, and the four input token pairs) and returns the attention output vectors for each of the four inputs.

## Canonical measurement condition

Every attempt **must** evaluate at the canonical slice `ρ = 0.7` and report the full sweep. The canonical dimension `d = 32` and matched Q/K are fixed. Attempts may not change the generator; they only provide `model_fn`.

## Model function contract

```python
def model_fn(batch: Batch) -> np.ndarray:
    """
    Args:
        batch: Batch with fields
            q_A: (d,), q_B: (d,)
            k_A: (d,), k_B: (d,)
            v_A: (d,), v_B: (d,)
            inputs: list[tuple[int, int]]  # four pairs: (0,0), (0,1), (1,0), (1,1)
    Returns:
        out: (4, d)  # attention output for each input in the same order as batch.inputs
    """
```

The attempt implements the forward pass of the 1-head attention block. The framework does **not** prescribe the implementation — the attempt may use any framework (PyTorch, JAX, NumPy) as long as `model_fn` is a pure Python callable with the above signature.

## Payload contract

`task.evaluate(model_fn)` returns a dict with the following exact keys:

```python
{
    "version": 1,                                    # int, mirrors benchmark.VERSION
    "config": {
        "d": 32,                                     # int
        "canonical_rho": 0.7,                        # float
        "rho_sweep": [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95],  # list[float]
    },
    "sweep": [
        {
            "rho": 0.0,                              # float
            "out_00": [...],                         # list[float] length d — attention output for (0,0)
            "out_01": [...],                         # list[float] length d — attention output for (0,1)
            "out_10": [...],                         # list[float] length d — attention output for (1,0)
            "out_11": [...],                         # list[float] length d — attention output for (1,1)
            "sharpness": 0.0,                        # float — see Metrics
        },
        ...                                          # one dict per rho in config.rho_sweep
    ],
}
```

- `out_*` are the first component of the attention output vector (the only non-zero component in the value basis). The full `d`-dim vector is included for completeness but metrics use only `out_*[0]`.
- `sharpness` is pre-computed by `task.evaluate` as the separation metric (defined below). `benchmark.score` recomputes it from `out_*` to verify consistency.

## Metrics

All metrics are **bigger-is-better** unless noted.

### Headline summary

| metric | formula | interpretation |
|--------|---------|----------------|
| `or_superposition_robustness` | `min_{ρ ∈ sweep} or_sharpness_ρ / or_sharpness_0p0` | Worst-case relative sharpness across the sweep. `1.0` = no degradation from interference; `0.0` = complete collapse at some ρ. |

### Per-slice values

For each `ρ` in the sweep, the tag is `f"{ρ:.2f}"` with `.`→`p` — i.e. `0p00`, `0p20`, `0p40`, `0p60`, `0p70`, `0p80`, `0p90`, `0p95`:

| metric | formula | interpretation |
|--------|---------|----------------|
| `or_sharpness_rho_<tag>` | `(mean(out_01, out_10, out_11)[0] - out_00[0]) / (max(out_01, out_10, out_11)[0] - min(out_01, out_10, out_11)[0] + ε)` | Separation of the `(0,0)` output from the `OR=1` outputs, normalised by the spread *within* the `OR=1` cluster (ε = 1e-8). Larger = cleaner separation; `≈0` = collapsed. Unbounded above when the three `OR=1` outputs are near-identical. |
| `or_gap_rho_<tag>` | `mean(out_01, out_10, out_11)[0] - out_00[0]` | Raw output gap (unnormalised). Useful for debugging scale. |

The canonical-slice values are also emitted as `or_sharpness_canonical` and `or_gap_canonical` (the `ρ = 0.7` values).

### Reference baselines

Reference for the **no-attention linear probe**. A linear map from the clean one-hot features `[A, B]` to the output sees no query interference, so its separation is `ρ`-independent and perfect on the four input pairs. `benchmark.score` therefore reports it as a fixed constant `1.0` at every slice (rather than re-solving the trivial probe each time):

| metric | value |
|--------|-------|
| `linear_baseline_sharpness_rho_<tag>` | `1.0` at every `ρ` (the linear probe's input is `ρ`-independent). |
| `linear_baseline_sharpness_canonical` | `1.0` (baseline at `ρ = 0.7`). |
| `lift_over_linear_canonical` | `or_sharpness_canonical - linear_baseline_sharpness_canonical`. |

## Bump procedure

- `VERSION` in `benchmark.py` **must** be bumped when:
  - Any metric formula changes.
  - Payload keys are added/removed/retyped.
  - Canonical `ρ` or `d` changes.
  - Sweep values change (add/remove/reorder).
- Update this README's "Payload contract" and "Metrics" tables in the same commit.
- Old `benchmark.json` files remain on disk; the dashboard filters to the highest `VERSION`.