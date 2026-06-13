# attention_mst

## Question

Given a **noisy** observation of a weighted graph over attention heads, can a
mechanism **recover the planted minimum spanning tree (MST)** — the backbone of
strongest pairwise relations — and does it keep recovering it as the
observation noise grows? The interesting claim is *denoising*: beating the
strawman that just runs Kruskal directly on the noisy weights.

## Setup

**Synthetic generator** — fully controlled, no trained models. For each
instance we sample a symmetric positive weight matrix over `n_heads` nodes
(edge weights ~ LogNormal(0, 1)), compute its true MST (Kruskal), then corrupt
the matrix with zero-mean Gaussian noise whose standard deviation is
`noise_level × median_edge_weight`. The attempt sees only the **noisy** matrix
and must score edges so that the predicted MST matches the planted one.

We sweep `noise_level ∈ {0.0, 0.1, …, 1.0}` to test robustness to corruption.

### Canonical measurement condition

- `n_heads = 12`
- noise sweep: `noise_level ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}`
- canonical noise = `0.5`
- `5` random seeds per noise level, averaged
- evaluation uses a fixed seed (`generate(seed=42)`); `generate` is
  deterministic for any given seed.

## Model function signature

The goal's contract with attempts. An attempt provides a `model_fn` and hands
it to `task.evaluate`; it never builds the payload itself.

```python
def model_fn(noisy_weights: np.ndarray) -> np.ndarray:
    """
    Args:
        noisy_weights: (n_heads, n_heads) symmetric, zero diagonal, the
                       NOISY observed edge weights.

    Returns:
        edge_scores: (n_heads, n_heads) symmetric edge scores. HIGHER score =
                     more likely to be an MST edge. (task.evaluate symmetrises
                     and zeroes the diagonal defensively, then runs Kruskal on
                     `-edge_scores` to extract the predicted MST.)
    """
```

`task.random_model_fn()` returns a reference `model_fn` that emits random
scores of the correct shape (used by the smoke test). The no-mechanism
**baseline** scoring `-noisy_weights` (run Kruskal directly on the noisy
observation) is computed internally by `task.evaluate` under identical
conditions.

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                       # int, matches benchmark.VERSION
    "model_name": "attention_mst",
    "n_heads": 12,                      # int
    "canonical_noise": 0.5,             # float, the canonical condition
    "noise_levels": [0.0, 0.1, ..., 1.0],   # list[float], the sweep axis
    "sweep": [                          # one record per noise_levels value
        {
            "noise_level": 0.5,         # float
            "edge_f1": 0.74,            # float in [0,1], MST edge-recovery F1
            "precision": 0.74,          # float in [0,1]
            "recall": 0.74,             # float in [0,1]
            "auroc": 0.91,              # float in [0,1], edge ranking
            "auprc": 0.65,              # float in [0,1]
            "weight_ratio": 1.08,       # float >= 1, pred MST weight / planted
            "n_seeds": 5,               # int
        },
        ...
    ],
    "baseline": [                       # same axis, no-mechanism reference
        { "noise_level": 0.5, "edge_f1": 0.5, "precision": ..., "recall": ...,
          "auroc": ..., "auprc": ..., "weight_ratio": ..., "n_seeds": 5 },
        ...
    ],
}
```

`sweep` and `baseline` are both lists the same length as `noise_levels`, each
indexed by its `noise_level` field. F1 / precision / recall / auroc / auprc are
in `[0, 1]` (higher is better); `weight_ratio ≥ 1` (lower is better, `1.0` is a
perfect-weight tree).

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars (floats `0.5` named
`0p5`):

| metric | meaning | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` (= 1) | — |
| `edge_f1_noise_0p0` … `edge_f1_noise_1p0` | per-noise MST edge-recovery F1 | **bigger = better** |
| `auroc_noise_0p0` … `_1p0` | per-noise edge-ranking AUROC | bigger = better |
| `weight_ratio_noise_0p0` … `_1p0` | per-noise pred/planted MST weight | smaller = better |
| `baseline_edge_f1_noise_0p0` … `_1p0` | baseline F1 per noise | reference |
| `baseline_auroc_noise_0p0` … `_1p0` | baseline AUROC per noise | reference |
| `edge_f1_canonical` | F1 at `canonical_noise` (0.5) | **bigger = better** |
| `auroc_canonical` | AUROC at `canonical_noise` (0.5) | bigger = better |
| `baseline_edge_f1_canonical` | baseline F1 at `canonical_noise` | reference |
| `lift_over_baseline_canonical` | `edge_f1_canonical − baseline_edge_f1_canonical` | bigger = better |
| `mst_recovery` | mean `edge_f1` across the sweep | **bigger = better** (headline) |
| `mst_recovery_robustness` | `edge_f1` at max noise (1.0) ÷ `edge_f1` at min noise (0.0), clipped to `[0,1]` | bigger = better |

### Headline summary

**`mst_recovery`** — the mean MST edge-recovery F1 across the whole noise
sweep. A mechanism that recovers the backbone across noise levels scores near
1.0; one that collapses under noise scores low. Read it alongside
`lift_over_baseline_canonical` (does the mechanism *denoise*, or just echo the
observed weights?) and `mst_recovery_robustness` (how gracefully it degrades).

## Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU (the smoke test runs
  `task`/`benchmark` on CPU/NumPy).
- `is_obviously_broken(metrics)` — short-circuits the jury when metrics are
  NaN/inf, or when `edge_f1_canonical` fails to beat
  `baseline_edge_f1_canonical` (no denoising at all).

## Bump procedure

Bump `VERSION` (in `benchmark.py` and this README, same commit) when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- `canonical_noise` or the sweep values change;
- a sweep/baseline record's schema changes.

Do **not** bump when adding a new metric that leaves existing ones unchanged,
or adding an optional payload key with a default. This goal is at `VERSION = 1`.
Old `benchmark.json` files stay on disk; the dashboard filters to the highest
version present.
