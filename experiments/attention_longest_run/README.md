# attention_longest_run

## Question
Given a transformer's attention weights, can we accurately measure the **longest consecutive run** of high attention weights to a target position across the sequence? This tests whether interpretability methods can recover a basic temporal property of attention patterns — the maximum span of contiguous positions that a head attends to above a threshold.

## Setup
**Synthetic generator only.** No trained model required. We generate sequences with implanted "run" patterns:
- A "target token" appears in a consecutive run of length `L` (1–16) at a random position.
- All other positions are random distractor tokens.
- The *true* longest run for a query at the last target token is exactly `L` (it attends back to the whole run).
- We simulate attention weights directly: for each head, we generate a weight vector where the run positions have weight `high` and others have weight `low`, plus Gaussian noise. Different heads have different `high`/`low` separations to create a difficulty sweep.

## Canonical measurement condition
- Sequence length: 64
- Vocabulary size: 128 (target token = 0, distractors 1–127)
- Run lengths `L` swept: `[1, 2, 3, 4, 5, 6, 8, 10, 12, 16]`
- 8 heads per sample, 128 samples per run length (1280 total)
- Attention threshold: `0.5` (weights > 0.5 count as "attending")
- Noise: `N(0, 0.15)` added to all weights, then clipped to `[0, 1]`
- Head difficulty parameter `d ∈ {0.3, 0.5, 0.7, 0.9}` controlling `high = 0.5 + 0.5*d`, `low = 0.5 - 0.5*d`
- Seed: 0 (fully deterministic; `seed` argument accepted but ignored)

## Model function signature
```python
def model_fn(tokens: np.ndarray,  # (batch, seq_len), int32
             attention_weights: np.ndarray) -> np.ndarray:  # (batch, n_heads, seq_len)
    """
    Return predicted longest-run length for each (batch, head).
    Output shape: (batch, n_heads), float32 or int32.
    """
    ...
```
The attempt receives both the tokens (to locate the target run) and the *ground-truth* attention weights with noise. This isolates the *measurement* problem: given noisy weights, can you compute the longest run above threshold?

## Payload contract
`task.evaluate` returns a dict with exactly these keys:
```python
{
    "version": 1,                          # int, matches benchmark.VERSION
    "canonical_threshold": 0.5,            # float, the threshold used
    "sweep": [
        {
            "run_length": 1,               # int, true run length L
            "difficulty": 0.3,             # float, head difficulty d
            "mae": 0.123,                  # float, mean absolute error of predicted vs true L
            "rmse": 0.156,                 # float, root mean squared error
            "correlation": 0.987,          # float, Pearson r (see note below)
            "n_samples": 256,              # int, number of (batch, head) pairs in this slice
        },
        ...                                # one record per (run_length, difficulty) pair
    ],
    "n_heads": 8,                          # int
    "seq_len": 64,                         # int
}
```
All floats are Python `float` (not numpy scalars). Each `(run_length, difficulty)`
slice covers the 128 samples at that run length × the 2 heads at that difficulty
= **256** predictions, so `mae` and `rmse` are computed over those 256 values and
`n_samples == 256`.

**Note on `correlation`:** within a single slice the true value is the constant
`L`, so a slice-local Pearson r would be undefined (zero variance). The
`correlation` field is instead computed **across all run lengths** for the heads
at that difficulty (predicted vs. true `L`, where `L` varies), and the same value
is attached to every record of that difficulty. Averaging `correlation` over run
lengths therefore reproduces that single per-difficulty value.

## Metrics
`benchmark.score` returns a flat dict:
| metric | formula | direction |
|--------|---------|-----------|
| `longest_run_mae_canonical` | MAE at canonical difficulty `d=0.5`, averaged over all run lengths | **smaller is better** |
| `longest_run_rmse_canonical` | RMSE at `d=0.5`, averaged over run lengths | **smaller is better** |
| `longest_run_corr_canonical` | Correlation at `d=0.5` (Pearson r of predicted vs. true `L` across run lengths; see payload note) | **bigger is better** |
| `longest_run_mae_d_<val>` | MAE at difficulty `d=val`, averaged over run lengths | **smaller is better** |
| `longest_run_rmse_d_<val>` | RMSE at difficulty `d=val`, averaged over run lengths | **smaller is better** |
| `longest_run_corr_d_<val>` | Correlation at difficulty `d=val` (Pearson r of predicted vs. true `L` across run lengths) | **bigger is better** |
| `longest_run_mae_L_<val>` | MAE at run length `L=val`, averaged over difficulties | **smaller is better** |
| `linear_baseline_mae_canonical` | MAE of a naive baseline (always predict mean run length) at `d=0.5` | **smaller is better** |
| `lift_over_linear_baseline_mae` | `linear_baseline_mae_canonical - longest_run_mae_canonical` | **bigger is better** |
| `version` | `benchmark.VERSION` | — |

Difficulty values in metric names use `0p3`, `0p5`, `0p7`, `0p9` format. Run lengths use plain integers.

## Bump procedure
- `VERSION` in `benchmark.py` must be incremented when any metric formula changes, a payload key is renamed/removed/retyped, or the canonical condition (threshold, seq_len, run-length sweep, difficulties) changes.
- Adding a new metric or a new difficulty slice does **not** require a bump.
- Update this README's "Payload contract" and "Metrics" tables in the same commit.