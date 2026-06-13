# attention_range_sum

## The question

Can a model (or a hand-built mechanism) read a contiguous **window** out of a
token sequence and report the **sum** of the values inside it — and how does
that ability degrade as the window grows?

This is a clean test of a range/aggregation circuit: an attention pattern that
must (a) select exactly the tokens in `[start, end)` and (b) accumulate their
values. Wider windows demand the attention to spread over more positions
without leaking onto out-of-window tokens, so MSE as a function of range length
is a sharp diagnostic.

## Setup

**Synthetic, fully deterministic.** `task.generate(seed)` builds one fixed
token sequence and a set of range-sum queries grouped by range length.

- `SEQ_LEN = 64` tokens, each an integer in `[0, VOCAB_SIZE)` with
  `VOCAB_SIZE = 10`.
- For each range length `k` in `RANGE_LENS = [2, 4, 8, 16, 32]`, the generator
  samples `QUERIES_PER_LEN = 200` windows: `start` uniform in `[0, L-k]`,
  `end = start + k`. The target is `sum(input_ids[start:end])`.

Same seed → same data. No I/O, no network.

## Canonical measurement condition

Every attempt must evaluate against `task.evaluate`, which internally calls
`generate(seed=42)` — the canonical batch. The canonical range length (the
slice used for the headline convenience metrics) is **`k = 8`**.

## The model_fn contract

An attempt hands `task.evaluate` a single callable:

```python
model_fn(input_ids: np.ndarray, start: int, end: int) -> float
```

- `input_ids` — the full sequence, shape `(64,)`, int values in `[0, 10)`.
- `start`, `end` — define the half-open window `[start, end)`.
- returns — a scalar prediction of `sum(input_ids[start:end])`.

The attempt never builds the payload itself; it passes the model and receives a
ready-to-record payload.

## Payload contract

`task.evaluate(model_fn)` returns:

| key       | type   | semantics |
|-----------|--------|-----------|
| `version` | `int`  | payload schema version (`1`); must equal `benchmark.VERSION` |
| `config`  | `dict` | generation config (`seq_len`, `vocab_size`, `range_lens`, `queries_per_len`, `seed`) — self-describing, not read by `score` |
| `sweep`   | `list` | one record per range length, in `RANGE_LENS` order |

Each `sweep` record:

| key           | type          | semantics |
|---------------|---------------|-----------|
| `range_len`   | `int`         | the window length `k` for this slice |
| `predictions` | `list[float]` | model outputs, one per query (length `QUERIES_PER_LEN`) |
| `targets`     | `list[float]` | ground-truth sums, same order/length |

`sweep` must have exactly `len(RANGE_LENS) == 5` records; `score` raises
`ValueError`/`KeyError` on any contract violation.

## Metrics

`benchmark.score(payload)` returns a flat dict (`version` first):

| metric | direction | meaning |
|--------|-----------|---------|
| `range_sum_robustness` | **headline**, bigger better, `[0,1]` | ratio of canonical-slice MSE to hardest-slice (`k=32`) MSE; `1.0` = no degradation as the range grows |
| `range_sum_mse_canonical` | smaller better | MSE at `k = 8` |
| `range_sum_mse_k_<k>` | smaller better | per-slice MSE for each `k` |
| `linear_baseline_mse_k_<k>` | reference | MSE of the optimal constant predictor (= variance of that slice's targets) |
| `lift_over_linear_k_8` | bigger better | `baseline − model` MSE at `k = 8`; positive means the model beats the no-mechanism floor |

The **baseline** is the best constant predictor per slice: a model that ignores
the window and emits the mean target. Beating it by a clear margin is the bar a
real range-sum mechanism must clear.

### Edge cases

- Empty slice → MSE/variance defined as `0.0`.
- Zero hardest-slice MSE → `range_sum_robustness = 1.0` (no division by zero).
- Non-finite or non-numeric payload entries → `ValueError`.

### `is_obviously_broken`

Returns `True` (skipping the expensive jury) when any metric is NaN/inf, or when
canonical MSE fails to beat the constant-predictor baseline by ≥10%
(`mse >= baseline * 0.9`). It never fires on a borderline-but-real result.

## Bump procedure

Bump `benchmark.VERSION` (currently `1`) and update this contract in the same
commit when you: change any existing metric's formula, rename/retype/remove a
payload key, or change the canonical range length, `RANGE_LENS`, or generator
parameters. Adding a new metric or an optional payload key does **not** require
a bump.

When you bump, also update the hardcoded `"version"` literal in
`task.evaluate` so the emitted payload still equals `benchmark.VERSION` —
otherwise `score` rejects every payload with a version mismatch.
