# attention_game_of_life

## Question

Can a model learn to compute one step of **Conway's Game of Life** — a local,
discrete, neighbourhood-counting cellular automaton — and predict the next
board state cell-by-cell? Game of Life is a clean stress test for a model that
must, at every cell, attend to its eight neighbours, count the live ones, and
apply a non-linear birth/survival rule. We measure how sharply an attempt
reproduces the true next state and how that holds up as board density varies.

## Setup

**Synthetic.** No trained-model dataset is required by the harness; the data
is generated deterministically in `task.py`. Boards are random binary grids of
shape `(B, H, W) = (32, 16, 16)`. The ground-truth next state is the exact
Game of Life update with **toroidal (wrap-around) boundaries** (`_next_state`).

Attempts supply a `model_fn` (typically a GPU model) that maps a board to
per-cell logits; the harness scores its predictions against the true update.

## Canonical measurement condition

The task sweeps the **initial live-cell density** of the boards over
`(0.1, 0.2, 0.3, 0.4, 0.5)`. The **canonical density is `0.3`** — headline
metrics are reported there. Boards are produced by `task.generate(seed=0)`;
a distinct RNG sub-stream is drawn per density so the slices are independent.

## `model_fn` contract

```python
model_fn(grids: np.ndarray) -> np.ndarray
```

| arg / return | shape       | dtype   | semantics                                            |
|--------------|-------------|---------|------------------------------------------------------|
| `grids` (in) | `(B, H, W)` | float32 | current board, values in `{0.0, 1.0}`                |
| return       | `(B, H, W)` | float   | per-cell **logit** that the cell is alive next step  |

A cell is predicted **alive** iff its returned logit is `> 0`. Output must be
finite and the same shape as the input, or `evaluate` raises `ValueError`.

Attempts never build the payload. They call `task.evaluate(model_fn)` and
receive a ready-to-record payload.

## Payload contract

`task.evaluate(model_fn)` returns:

| key                 | type          | semantics                                              |
|---------------------|---------------|--------------------------------------------------------|
| `version`           | int           | payload schema version (`1`)                           |
| `grid_size`         | int           | `H` (= `W`)                                             |
| `batch_size`        | int           | `B`                                                    |
| `seed`              | int           | generator seed used (`0`)                              |
| `canonical_density` | float         | `0.3`                                                  |
| `density_sweep`     | list[float]   | the densities measured, in order                       |
| `sweep`             | list[record]  | one record per density (same length as `density_sweep`)|

Each `sweep` record (all counts are integers, pre-aggregated — no tensors):

| field            | meaning                                                         |
|------------------|-----------------------------------------------------------------|
| `density`        | initial live-cell fraction for this slice                       |
| `n_cells`        | total cells scored (`B*H*W`)                                     |
| `n_correct`      | cells whose predicted alive/dead matched the truth              |
| `tp,fp,fn,tn`    | confusion counts for the *alive-next* positive class            |
| `static_correct` | cells correct under the static baseline (predict next = current)|
| `static_tp,static_fp,static_fn,static_tn` | static baseline confusion counts       |

## Metrics

`benchmark.score(payload)` returns a flat dict. All metrics are
**bigger-is-better**. `version` is the first key; the dashboard filters to the
highest version present.

| metric                                   | meaning                                                        |
|------------------------------------------|----------------------------------------------------------------|
| `life_robustness` *(headline)*           | worst-case `next_state_f1` across the density sweep, in `[0,1]` |
| `next_state_f1_canonical`                | alive-next F1 at the canonical density (`0.3`)                  |
| `next_state_accuracy_canonical`          | cell accuracy at the canonical density                         |
| `lift_over_static_f1_canonical`          | canonical F1 minus the static-baseline F1                      |
| `mean_next_state_f1`                     | mean F1 across the sweep                                        |
| `mean_next_state_accuracy`               | mean cell accuracy across the sweep                            |
| `next_state_accuracy_density_<d>`        | per-slice cell accuracy (`<d>` as `0p3`)                        |
| `next_state_f1_density_<d>`              | per-slice alive-next F1                                         |
| `live_recall_density_<d>`                | per-slice recall of the alive-next class                       |
| `live_precision_density_<d>`             | per-slice precision of the alive-next class                    |
| `static_baseline_accuracy_density_<d>`   | per-slice accuracy of predicting next = current                |
| `static_baseline_f1_density_<d>`         | per-slice F1 of predicting next = current                      |

**Reading them.** Cell accuracy is dominated by dead cells (most cells stay
dead), so the **F1 of the alive-next class** is the meaningful signal — it
rewards correctly predicting births and survivals. The static baseline ("copy
the board") is measured under identical boards: a real method must clear it
(`lift_over_static_f1_canonical > 0`). `life_robustness` is the single number
to optimise — high F1 at *every* density, not just the easy ones.

### Degenerate-run hook

`is_obviously_broken(metrics)` returns `True` (skipping the jury) when any
metric is NaN/inf, or when
`next_state_f1_canonical <= static_baseline_f1_density_0p3` (the attempt does
not beat the trivial copy baseline).

## Bump procedure (`VERSION`)

`benchmark.VERSION = 1`. Bump it when you change the formula of an existing
metric, rename/retype a payload key, or move the canonical density. Adding a
new metric or a new density slice (the sweep is length-matched but otherwise
extensible) does **not** require a bump. When you bump, update this contract in
the same commit; old `benchmark.json` files stay on disk and are filtered out
by the dashboard.
