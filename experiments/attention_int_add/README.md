# attention_int_add

## The question

Multi-digit integer addition has an easy part and a hard part. The easy part
is column-wise digit addition; the hard part is **carry propagation**, where a
carry generated in one column must be routed into the next-more-significant
column. For an attention-based model this routing is exactly the kind of
inter-position information movement attention is supposed to provide.

This goal asks: **how well does an attempt's model compute integer addition,
and how robust is that computation as the number of carries grows?** A method
that aces zero-carry problems but collapses on full carry chains has not
learned addition — it has learned a per-column lookup table.

## Setup

Fully **synthetic**. `task.generate` produces addition problems `a + b` with
`a, b` in `[0, 999]` (`MAX_DIGITS = 3`), bucketed into slices by the exact
number of carries (`0, 1, 2, 3`). Each slice holds `NUM_SAMPLES_PER_SLICE`
(=300) problems, drawn deterministically from a seeded pool. `seed` is honored;
the canonical batch uses `seed=0`.

There is no trained model shipped with the goal — the attempt supplies the
model. Attempts may train a tiny transformer, hand-construct weights, or
anything else, as long as they expose the `model_fn` contract below.

## Canonical measurement condition

- `MAX_DIGITS = 3`, `SUM_DIGITS = 4`, `VOCAB_SIZE = 15`, `SEQ_LEN = 14`.
- Batch from `generate(seed=0)`.
- The **canonical condition** is `carries = 3` (the full carry chain), the
  hardest slice. The headline and the `*_canonical` metrics report it.

## Sequence layout & vocabulary

Token IDs `0..9` are literal digits. Special tokens:

| token | id |
|-------|----|
| `+`   | 10 |
| `=`   | 11 |
| BOS   | 12 |
| EOS   | 13 |
| PAD   | 14 |

A sequence (most-significant digit first, zero-padded):

```
idx:  0    1 2 3   4    5 6 7   8     9 10 11 12   13
tok: BOS  a a a   +    b b b   =    PAD PAD ...    EOS
```

The four SUM positions (`idx 9..12`) are **masked to PAD on input** — the model
must predict them.

## The `model_fn` contract

```python
model_fn(input_ids: np.ndarray[int]  shape (N, SEQ_LEN))
    -> logits: np.ndarray[float]      shape (N, SEQ_LEN, VOCAB_SIZE)
```

`evaluate` reads `argmax(logits, axis=-1)` at the four SUM positions only and
compares those predicted digits against the true sum digits. The model never
constructs the payload; it just returns logits.

## Payload contract

`task.evaluate(model_fn)` returns:

| key                 | type                | meaning |
|---------------------|---------------------|---------|
| `version`           | `int` (== 1)        | payload/benchmark version |
| `max_digits`        | `int`               | operand digit width (3) |
| `sum_digits`        | `int`               | sum digit width (4) |
| `canonical_carries` | `int`               | canonical slice (3) |
| `carry_sweep`       | `list[int]`         | slice axis, `[0,1,2,3]` |
| `sweep`             | `list[record]`      | one record per carry slice |
| `linear_baseline`   | `list[record]`      | no-carry strawman, same slices |

Each `sweep` / `linear_baseline` record:

| field              | type    | meaning |
|--------------------|---------|---------|
| `carries`          | `int`   | number of carries for this slice |
| `exact_match_rate` | `float` | fraction of problems with all 4 sum digits correct |
| `digit_accuracy`   | `float` | fraction of individual sum digits correct |
| `n`                | `int`   | number of problems in the slice |

The **linear baseline** is model-independent: it adds each column mod 10 and
ignores all carries (leading sum digit predicted 0). It is the no-mechanism
reference — beating it is what makes a result meaningful.

## Metrics (`benchmark.score`)

`version` is the first key. All rates are in `[0, 1]`, bigger-is-better.

| metric                                    | meaning |
|-------------------------------------------|---------|
| `carry_robustness` *(headline)*           | `exact_match(carries=3) / exact_match(carries=0)`, clipped to `[0,1]`. The single number to optimise. |
| `exact_match_carries_<k>`                 | per-slice exact-match rate |
| `digit_accuracy_carries_<k>`              | per-slice digit accuracy |
| `linear_baseline_exact_match_carries_<k>` | baseline exact-match, same slice |
| `exact_match_canonical`                   | exact-match at `carries=3` |
| `lift_over_baseline_canonical`            | `exact_match_canonical` − baseline at `carries=3` |
| `exact_match_mean`                        | unweighted mean of per-slice exact-match |

### Edge cases

- Empty / short slices contribute `0.0` rather than dividing by zero.
- `carry_robustness` is `0.0` when the zero-carry rate is `≈ 0` (avoids `inf`).
- `score` raises `ValueError` / `KeyError` on any contract violation.

### `is_obviously_broken`

Returns `True` (skip the jury) when any metric is NaN/inf, or when
`exact_match_canonical` fails to beat the linear baseline at `carries=3`. It
never returns `True` for a borderline-but-real result.

## Bump procedure

Bump `VERSION` (in `benchmark.py`, the payload, and this table) when you change
any existing metric formula, rename/retype a payload key, or move the canonical
condition. Adding a new metric or an extensible slice does **not** require a
bump. Old `benchmark.json` files stay on disk; the dashboard filters to the
highest version.
