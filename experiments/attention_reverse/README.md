# attention_reverse

## Question

Can a transformer's attention heads implement **sequence reversal** —
predicting, at every position `i`, the token sitting at the mirror position
`L-1-i` — and does that mechanism **generalise to longer sequences** than the
canonical length?

Reversal is the cleanest non-induction positional algorithm: it needs a head
whose query at position `i` attends to key position `L-1-i`. We score both the
**behaviour** (does the model output the reversed sequence?) and the
**mechanism** (does the relevant head actually place its attention mass on the
mirror position?).

## Setup

Synthetic. Sequences of i.i.d. uniform token ids are drawn from a fixed
vocabulary; the target at position `i` is the input token at the mirror
position `L-1-i`. Attention is **bidirectional (no causal mask)** — position
`0` must predict the final token, which lies in the future, so a causal mask
would make the task unsolvable. All data is generated deterministically from a
seed in `task.py`; no I/O, no network, no trained-model dependency.

## Canonical measurement condition

| parameter        | value           |
|------------------|-----------------|
| vocabulary size  | 16              |
| canonical length | 16              |
| length sweep     | 8, 16, 32, 64   |
| sequences/slice  | 256             |
| eval batch size  | 32              |
| attention mask   | bidirectional   |
| eval seed        | 42              |

The canonical slice is `seq_len = 16` (`canonical_idx = 1`). Lengths `32` and
`64` are the **generalisation** slices used by the headline metric.

## model_fn contract

An attempt hands `task.evaluate` a single callable:

```python
model_fn(tokens) -> (logits, attn)
    tokens : np.ndarray (batch, seq_len)          int   token ids in [0, vocab)
    logits : np.ndarray (batch, seq_len, vocab)   float per-position prediction
    attn   : np.ndarray (batch, seq_len, seq_len) float attention pattern;
             attn[b, i, j] = weight query position i places on key position j
```

- `logits` drive the **behavioural** metric: `argmax(logits, axis=-1)` is the
  predicted reversed sequence.
- `attn` drives the **mechanistic** metric: the mass `attn[b, i, L-1-i]` placed
  on each mirror position. Rows need not be pre-normalised — `evaluate`
  clips negatives and re-normalises each row over keys defensively. If a model
  has no single attention head, return that head's pattern (or a uniform
  pattern); the metric simply reports near-`1/L` mass in that case.

Attempts never build the payload themselves: they pass `model_fn` to
`task.evaluate`, which returns a ready-to-record payload.

## Payload contract

`task.evaluate(model_fn)` returns:

| key              | type              | meaning                                        |
|------------------|-------------------|------------------------------------------------|
| `version`        | int (== 1)        | payload schema version                         |
| `model_name`     | str               | label, not read by `score`                     |
| `config`         | dict              | full canonical config (self-describing)        |
| `vocab_size`     | int               | vocabulary size (for the random baseline)      |
| `seq_len_sweep`  | list[int]         | sequence lengths, ascending                    |
| `canonical_idx`  | int               | index of the canonical length within the sweep |
| `sweep`          | list[record]      | one record per length (see below)              |

Each `sweep` record:

| key                          | type  | meaning                                      |
|------------------------------|-------|----------------------------------------------|
| `seq_len`                    | int   | sequence length for this slice               |
| `accuracy`                   | float | fraction of positions predicted correctly    |
| `mirror_attn_mass`           | float | mean attention mass on the mirror position    |
| `identity_baseline_accuracy` | float | data-driven no-reversal (same-position) acc.  |
| `num_sequences`              | int   | sequences evaluated in this slice            |

## Metrics

All produced by `benchmark.score(payload)`. `version` is first; the dashboard
filters to the highest version present. Direction: **bigger is better** for
every metric here.

| metric                                | meaning                                                            |
|---------------------------------------|-------------------------------------------------------------------|
| `length_generalization_robustness`    | **headline.** `min(acc at lengths > canonical) / acc_canonical`, clamped to `[0,1]`. |
| `reverse_accuracy_canonical`          | accuracy at the canonical length (16).                            |
| `reverse_accuracy_len_<L>`            | per-slice accuracy at length `L`.                                 |
| `mirror_attn_mass_canonical`          | mean mirror attention mass at the canonical length.               |
| `mirror_attn_mass_len_<L>`            | per-slice mirror attention mass at length `L`.                    |
| `identity_baseline_accuracy_len_<L>`  | per-slice no-reversal reference accuracy at length `L`.           |
| `random_baseline_accuracy`            | `1 / vocab_size` — chance level.                                  |
| `lift_over_random_canonical`          | `reverse_accuracy_canonical - random_baseline_accuracy`.          |

**How to read.** A real reversal head pushes `reverse_accuracy_*` and
`mirror_attn_mass_*` toward 1 and far above `random_baseline_accuracy`. The
headline `length_generalization_robustness` separates models that memorised a
length-16 lookup (high canonical accuracy, robustness near 0) from models that
learned the positional algorithm (robustness near 1).

### Edge cases

- Empty generalisation set or zero canonical accuracy → robustness `0.0`.
- Zero attention rows are renormalised to uniform before measuring mass.
- Any NaN/inf, or canonical accuracy at-or-below the random baseline, is flagged
  by `is_obviously_broken` so the jury is skipped.

## Pipeline hooks

- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken(metrics)` → `True` on NaN/inf or when
  `reverse_accuracy_canonical <= 1.05 * random_baseline_accuracy`.

## Bump procedure

Bump `VERSION` (currently `1`) when changing any existing metric formula,
renaming/retyping a payload key, or changing the canonical condition (vocab,
canonical length, sweep, mask). Adding a new metric or an optional payload key
does not require a bump. Update this contract in the same commit.
