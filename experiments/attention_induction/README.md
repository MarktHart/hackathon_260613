# attention_induction

## Question

Does the model implement an **induction head** — given a token that appeared
earlier in the context, does it predict *the token that followed that earlier
occurrence*? Concretely, for a duplicated pattern block `[A_0 … A_{P-1}][A_0 …
A_{P-1}]`, at a second-copy position holding `A_j` a correct induction
mechanism predicts `A_{j+1}`. We sweep the occurrence **distance** `P` to test
how well the mechanism copies from progressively further back.

## Setup

**Synthetic generator** — fully controlled, no trained dataset. Each sequence
is random background tokens with one duplicated "pattern" block injected at a
random offset:

```
[ ... background ... ][ A_0 A_1 ... A_{P-1} ][ A_0 A_1 ... A_{P-1} ][ ... background ... ]
```

To keep ground truth unambiguous:

- **Pattern tokens** are drawn *without replacement* from `[0, 64)`, so each
  pattern token's first occurrence is unique.
- **Background tokens** are drawn from the *disjoint* range `[64, 128)`, so
  background never collides with pattern tokens.

The induction target at second-copy position `offset + P + j` (holding `A_j`)
is the token `A_{j+1}`, for `j ∈ {0, …, P-2}`. The two occurrences of `A_j`
are exactly `P` apart — that separation is the swept distance.

### Canonical measurement condition

- `vocab_size = 128` (pattern `[0,64)`, background `[64,128)`)
- `seq_len = 192`
- distance sweep `P ∈ {16, 32, 48, 64}` (one bucket each)
- `16` sequences per bucket → `batch_size = 64`
- **canonical distance = 16** (shortest separation, easiest)
- fixed seed `42` — `generate()` accepts a `seed` arg but the canonical
  condition always uses `42`, so every attempt evaluates on identical data.

## Model function signature

```python
def model_fn(input_ids: np.ndarray) -> np.ndarray:
    """
    Args:
        input_ids: (batch, seq_len) int — token ids in [0, vocab_size)
    Returns:
        logits: (batch, seq_len, vocab_size) float — next-token logits;
                logits[b, p] scores the token that follows position p.
    """
```

The attempt returns raw logits; `task.evaluate` reads logits at the induction
target positions, applies softmax, and computes accuracy / cross-entropy.

## Payload contract

`task.evaluate(model_fn)` returns exactly:

```python
{
    "version": 1,                          # matches benchmark.VERSION
    "model_name": "synthetic_induction",   # fixed identifier
    "vocab_size": 128,
    "seq_len": 192,
    "canonical_distance": 16,
    "sweep": [                             # one record per distance bucket
        {
            "distance": 16,                # occurrence separation P
            "num_targets": 240,            # induction targets in this bucket
            "accuracy": 0.91,              # fraction predicted correctly, [0,1]
            "ce_loss": 0.42,               # mean cross-entropy (nats), >= 0
            "uniform_baseline_accuracy": 0.0078125,   # 1 / vocab_size
            "uniform_baseline_ce_loss": 4.852,        # log(vocab_size)
        },
        ...                                # distances 32, 48, 64
    ],
    "aggregate": {                         # over all valid targets
        "accuracy": 0.85,
        "ce_loss": 0.61,
        "num_targets": 2496,               # 16 * sum(P-1 for P in {16,32,48,64})
        "uniform_baseline_accuracy": 0.0078125,
        "uniform_baseline_ce_loss": 4.852,
    },
}
```

Accuracy is in `[0,1]` (bigger = better); `ce_loss` is in nats (smaller =
better).

## Metrics

`benchmark.score(payload)` returns a flat dict:

| metric | meaning | direction |
|--------|---------|-----------|
| `version` | `benchmark.VERSION` | — |
| `induction_accuracy` | overall accuracy across all targets | **bigger = better (headline)** |
| `induction_ce_loss` | overall mean cross-entropy | smaller = better |
| `induction_accuracy_canonical` | accuracy at the canonical distance (16) | bigger = better |
| `induction_accuracy_dist_16` … `_dist_64` | per-distance accuracy | bigger = better |
| `induction_ce_loss_dist_16` … `_dist_64` | per-distance cross-entropy | smaller = better |
| `num_targets_dist_16` … `_dist_64` | target count per bucket | — |
| `uniform_baseline_accuracy` | `1 / vocab_size` reference | reference |
| `uniform_baseline_ce_loss` | `log(vocab_size)` reference | reference |
| `lift_over_uniform` | `induction_accuracy − uniform_baseline_accuracy` | bigger = better |
| `distance_robustness` | accuracy at max distance ÷ accuracy at min distance, clipped to `[0,1]` | bigger = better |
| `num_targets` | total valid targets | — |

### Headline summary

**`induction_accuracy`** — the single number an attempt optimises: the fraction
of induction targets predicted correctly across the whole sweep. The uniform
baseline is `1/128 ≈ 0.0078`; anything near that has no induction signal.
`distance_robustness` is the secondary read on whether the mechanism degrades
as the copy distance grows.

## Edge cases

- Empty `sweep` or missing `aggregate`/`version`/`canonical_distance` → `score`
  raises `ValueError`/`KeyError`.
- A distance bucket with zero targets reports accuracy/ce `0.0`.
- `distance_robustness` returns `0.0` when the nearest-distance accuracy is `0`
  (no division by zero).

## Bump procedure

Bump `VERSION` in `benchmark.py` and this README together when:

- any metric formula changes;
- a payload key is added/removed/renamed or retyped;
- the canonical distance, distance sweep, vocab split, or sequence length
  changes.

Do **not** bump when adding a new metric that leaves existing ones untouched,
or adding an optional payload key with a default. Old `benchmark.json` files
stay on disk; the dashboard filters to the highest version present.
