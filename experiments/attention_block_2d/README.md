# attention_block_2d

## Question

Can an interpretability method recover the **spatial attention pattern** (local window size, dilation, global-token presence) of a 2D attention block from its attention weight matrix alone?

## Setup

**Synthetic generator only.** No trained model, no dataset. Each example is a single-head attention weight matrix `A ∈ ℝ^(N×N)` where `N = H·W` is the number of 2D positions (default `H = W = 8`, so `N = 64`). The matrix is generated from a known **pattern family** with ground-truth parameters:

| pattern_id | description | parameters |
|------------|-------------|------------|
| `local`    | square window around each query position | `window_size ∈ {1,2,3}` (radius; `window_size=1` → 3×3 window) |
| `dilated`  | window with stride > 1 | `window_size ∈ {1,2}`, `dilation ∈ {1,2}` |
| `global`   | one global token attends to all / all attend to global | `global_pos ∈ {0, N-1}` (index of global token) |
| `causal_2d`| raster-order causal (each position attends to earlier positions) | none |

The generator constructs a *row-stochastic* matrix (rows sum to 1) by placing uniform mass over the allowed keys for each query, then adding i.i.d. uniform noise in `[0, ε]` (`ε = 1e-3`, drawn from the seeded RNG) and renormalising. The noise makes each of the 4 realisations per pattern distinct and seed-dependent.

## Canonical measurement condition

- Grid: `8 × 8` (`N = 64`)
- Patterns: all four families, each at their canonical parameter values  
  (`local: window_size=1`, `dilated: window_size=1,dilation=2`, `global: global_pos=0`, `causal_2d`)
- Seed: `0` (deterministic batch of 16 matrices, 4 per pattern)

## Model function signature

```python
def model_fn(attn: np.ndarray) -> dict:
    """
    Parameters
    ----------
    attn : np.ndarray, shape (N, N)
        Row-stochastic attention matrix for a single head.

    Returns
    -------
    dict with keys:
        "pattern_id"   : str   — one of {"local","dilated","global","causal_2d"}
        "params"       : dict  — predicted parameters for that pattern family
                           (keys match the table above; extra keys ignored)
        "confidence"   : float — ∈ [0, 1], self-reported confidence
    """
```

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                          # matches benchmark.VERSION
    "grid_size": (8, 8),                   # (H, W)
    "sweep": [
        {
            "pattern_id": "local",
            "params": {"window_size": 1},
            "pred_pattern_id": "local",
            "pred_params": {"window_size": 1},
            "confidence": 0.92,
            "correct": true
        },
        ... (16 records total)
    ]
}
```

- `sweep` length is always 16 (4 patterns × 4 noise realisations each).
- `correct` is `true` iff `pred_pattern_id == pattern_id` **and** all
  parameter values match exactly (for `global`, `global_pos` must match).
- `confidence` is passed through from `model_fn` unchanged.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| metric | formula | bigger is better? |
|--------|---------|-------------------|
| `version` | `payload["version"]` | — |
| `pattern_acc_canonical` | fraction of 16 records with `correct == true` | ✓ |
| `pattern_acc_local` | accuracy on the 4 `local` records | ✓ |
| `pattern_acc_dilated` | accuracy on the 4 `dilated` records | ✓ |
| `pattern_acc_global` | accuracy on the 4 `global` records | ✓ |
| `pattern_acc_causal_2d` | accuracy on the 4 `causal_2d` records | ✓ |
| `mean_confidence_correct` | mean `confidence` over correct predictions | ✓ |
| `mean_confidence_incorrect` | mean `confidence` over incorrect predictions | ✗ (lower better) |
| `linear_baseline_acc` | accuracy of a fixed majority-class baseline (always predicts `"local"`) | — |
| `lift_over_linear_baseline` | `pattern_acc_canonical - linear_baseline_acc` | ✓ |

All accuracies in `[0, 1]`. Confidences in `[0, 1]`.

## Bump procedure

- `VERSION` increments when: payload keys change, canonical grid size changes,
  pattern families/parameters change, or any metric formula changes.
- Adding a new per-pattern accuracy metric does **not** require a bump.
- After bump: update this README's "Payload contract" and "Metrics" tables
  in the same commit.