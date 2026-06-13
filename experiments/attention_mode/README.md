# Goal: attention_mode

## Question

Given a transformer head's raw attention matrix, can an interpretability
mechanism **name the canonical mode** the head implements — and does that
judgement degrade gracefully as the pattern is corrupted by noise?

A "mode" is a human-nameable attention pattern. This goal uses five:

| mode             | description                                          |
|------------------|------------------------------------------------------|
| `positional`     | every query attends to one fixed key position        |
| `uniform`        | uniform attention over all keys                      |
| `diagonal`       | each query attends to its own index (i → i)          |
| `induction`      | each query attends to the next position (i → i+1)    |
| `previous_token` | each query attends to the previous position (i → i-1)|

## Setup

**Fully synthetic.** `task.generate(seed, noise)` builds clean attention
matrices for each mode and mixes in a controlled amount of uniform noise:

    out = (1 - noise) * clean_pattern + noise * random_row_stochastic

Rows always sum to 1. For each mode it emits `N_PER_MODE = 10` heads, so a
batch at one noise level holds `10 × 5 = 50` heads.

No real model and no dataset are involved — the patterns *are* the ground
truth. This keeps two attempts from ever disagreeing about the data.

## Canonical measurement condition

- sequence length `L = 16`
- `seed = 0`
- noise sweep `NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]`
- **canonical noise = 0.0** (clean patterns)

`evaluate` runs the attempt's `model_fn` over **every** noise level and
records one sweep entry per head.

## `model_fn` contract

```python
model_fn(attention_matrices: np.ndarray) -> np.ndarray
```

- **input**  `(n_heads, L, L)` float32; each row of each `(L, L)` matrix is a
  probability distribution over keys (sums to 1).
- **output** `(n_heads, N_MODES)` float32; each row a probability distribution
  over the modes **in `MODES` order**
  (`positional, uniform, diagonal, induction, previous_token`), summing to 1.

`evaluate` normalises the rows defensively and raises `ValueError` on wrong
shape, non-finite values, negative entries, or all-zero rows.

`task.random_model_fn()` returns a reference `model_fn` that emits uniform
random rows — used by the pipeline smoke test and as the conceptual baseline.

## Payload contract

`task.evaluate(model_fn)` returns, and `benchmark.score(payload)` consumes:

| key              | type                  | semantics                                   |
|------------------|-----------------------|---------------------------------------------|
| `version`        | int                   | payload schema version (must equal `VERSION`)|
| `L`              | int                   | sequence length (16)                        |
| `seed`           | int                   | generation seed (0)                         |
| `modes`          | list[str]             | the five mode names, in `MODES` order       |
| `noise_levels`   | list[float]           | noise levels swept                          |
| `canonical_noise`| float                 | the canonical slice (0.0)                   |
| `n_per_mode`     | int                   | heads per mode per noise level (10)         |
| `sweep`          | list[record]          | one entry per head per noise level          |

Each `sweep` record:

| field        | type             | semantics                                  |
|--------------|------------------|--------------------------------------------|
| `noise`      | float            | noise level for this head                  |
| `true_mode`  | str              | ground-truth mode (in `modes`)             |
| `pred_probs` | dict[str, float] | predicted probability per mode (keys == `modes`) |

The payload carries **no tensors** — only scalars and small dicts.

## Metrics

`benchmark.score` returns a flat dict (`version` first). Direction: **bigger
is better** for every metric except `cross_entropy_*` (smaller is better).

| metric                                | meaning                                                        |
|---------------------------------------|----------------------------------------------------------------|
| `mode_robustness` *(headline)*        | accuracy at hardest noise ÷ canonical accuracy, clamped `[0,1]` |
| `accuracy_canonical`                  | top-1 accuracy on clean patterns                               |
| `macro_f1_canonical`                  | macro-averaged F1 on clean patterns                            |
| `cross_entropy_canonical`             | mean −log P(true mode) on clean patterns (lower is better)     |
| `accuracy_mode_<mode>`                | per-mode canonical accuracy                                    |
| `accuracy_noise_<v>`                  | per-slice accuracy; `v` in `0p0`-form                          |
| `linear_baseline_accuracy_canonical`  | random-guess reference (`1/N_MODES = 0.2`)                     |
| `lift_over_baseline_accuracy`         | `accuracy_canonical − baseline`                                |

**Headline = `mode_robustness`.** It rewards mechanisms that both classify
clean patterns *and* stay correct as patterns are corrupted. Inspect
`accuracy_noise_*` to see where a method breaks.

### Pipeline hooks

- `GPU_REQUIREMENT = 1` — attempts run on the GPU; `task`/`benchmark` are CPU-only.
- `is_obviously_broken(metrics)` — `True` on NaN/inf, or when
  `accuracy_canonical ≤ 1.5 × baseline` (no better than near-random). Skips the
  jury on clearly degenerate attempts; never fires on borderline-real results.

## Bump procedure

Bump `VERSION` (in `benchmark.py`) and update this contract in the same commit
when you: change a metric formula, rename/retype/remove a payload key, or move
the canonical condition (e.g. `L`, the noise sweep, or `canonical_noise`).
Adding a new metric or an extra noise slice does **not** require a bump. Old
`benchmark.json` files stay on disk; the dashboard filters to the highest
version present.
