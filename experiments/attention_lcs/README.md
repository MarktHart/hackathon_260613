# attention_lcs

## Question

Do attention heads track the **Longest Common Subsequence (LCS)** alignment
between two sequences? When a model reads sequence A while looking at sequence
B, does any head concentrate its attention on the key positions in B that are
the LCS-match partners of each query position in A?

This is a soft test for *alignment-tracking* circuitry: an LCS alignment is a
monotone, non-crossing matching of equal symbols. A head that implements it
would put most of a query's attention mass on that query's matched key.

## Setup

**Synthetic generator.** We sample pairs of sequences over a small vocabulary
and compute, by dynamic programming, one canonical LCS alignment for each
pair. The alignment yields, for every query position in A that participates in
the LCS, the set of key positions in B it matches.

The generator is fully deterministic given a seed.

### Canonical measurement condition

Every attempt must evaluate on the batch produced by `generate()` with the
defaults below (this is what `evaluate()` uses internally):

- `seq_len = 16`
- `vocab_size = 8`
- `num_examples = 256`
- `seed = 0`

A smaller vocabulary than uniform-random text guarantees most pairs have a
non-trivial LCS (several matched positions), so the metric is well populated.

## Model function signature

This is the goal's contract with attempts. An attempt hands `evaluate` a
callable of exactly this shape:

```python
def model_fn(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
    """
    Args:
        seq_a: int32 array [batch, seq_len] — query sequence (A).
        seq_b: int32 array [batch, seq_len] — key sequence (B).

    Returns:
        attn: float32 array [batch, n_heads, seq_len, seq_len].
              Cross-attention from each A position (query, axis -2) to each
              B position (key, axis -1). Must be finite, non-negative, and
              sum to 1 over the last axis (keys) for every query.
    """
```

`n_heads` is chosen by the attempt (any value ≥ 1). The evaluator reads it
from the returned array's shape.

## Payload contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                       # int — must equal benchmark.VERSION
    "config": {                         # dict — self-describing, echoed for debugging
        "seq_len": 16,
        "vocab_size": 8,
        "num_examples": 256,
        "seed": 0,
        "n_heads": 4,                   # int — heads the attempt produced
    },
    "random_baseline_mass": 0.0625,     # float — expected LCS mass under uniform attention
    "sweep": [                          # list[record], one per head, ordered by head index
        {
            "head": 0,                  # int
            "lcs_attention_mass": 0.42, # float in [0,1] — mean attention mass on LCS keys
            "lcs_lift": 0.3575,         # float — lcs_attention_mass - random_baseline_mass
            "n_query_positions": 1696,  # int — total LCS query positions scored for this head
        },
        ...
    ],
}
```

Semantics:

- `lcs_attention_mass`: for every query position in A that has at least one
  LCS-match key in B, sum the head's attention weights on those match keys;
  average over all such query positions across the whole batch.
- `random_baseline_mass`: the mass a *uniform* head would place on the same
  keys, i.e. `mean over scored queries of (#match_keys / seq_len)`. It does
  not depend on the model, only on the data, so it is reported once at the top
  level rather than per head.
- `lcs_lift`: `lcs_attention_mass - random_baseline_mass`, a per-head signal
  of how much better than chance that head aligns.
- `n_query_positions`: bookkeeping; lets the grader see coverage.

## Metrics

`benchmark.score(payload)` returns a flat dict of scalars:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `version` | `payload["version"]` | — | Benchmark version (always first key) |
| `random_baseline_mass` | from payload | bigger = denser LCS | Uniform-attention reference mass |
| `lcs_attention_head_<h>` | per-head `lcs_attention_mass` | **bigger better** | Each head's LCS attention mass |
| `lcs_lift_head_<h>` | per-head `lcs_lift` | **bigger better** | Each head's lift over uniform |
| `lcs_attention_canonical` | `max_h lcs_attention_mass` | **bigger better** | Best head's absolute mass |
| `lcs_lift_canonical` | `max_h lcs_lift` | **bigger better** | **Headline.** Best head's lift over uniform |
| `lcs_robustness` | `clip(lcs_lift_canonical / (1 - random_baseline_mass), 0, 1)` | **bigger better** | Lift as a fraction of the available headroom above chance |

### Headline summary

**`lcs_lift_canonical`** — the best head's mean attention mass on LCS keys
*minus* the uniform-attention baseline. A head that ignores the alignment
scores ≈ 0; a head that puts all mass on LCS keys scores ≈ `1 -
random_baseline_mass`. This is the single number an attempt optimises.

`lcs_robustness` normalises that lift by the maximum achievable lift
(`1 - random_baseline_mass`) so it lands in `[0, 1]` and is comparable across
data regimes.

### Per-slice values

One metric per head — `lcs_attention_head_0`, `lcs_lift_head_0`, … — so the
dashboard's dropdown lets the grader see which heads (if any) carry the
alignment and which are doing something else.

### Reference baseline

`random_baseline_mass` is the no-mechanism reference under identical data:
exactly the mass uniform attention would place on LCS keys. Every per-head
`lcs_lift` is measured against it.

## Bump procedure

Bump `benchmark.VERSION` (and update this contract in the same commit) when:

- a payload key is added, removed, or retyped;
- any metric formula changes;
- the canonical condition (`seq_len`, `vocab_size`, `num_examples`, `seed`)
  changes.

You need **not** bump when adding a new metric without touching existing ones.

## Optional pipeline hooks

- `GPU_REQUIREMENT = 1` (every attempt runs on GPU).
- `is_obviously_broken(metrics)` returns `True` when any metric is NaN/inf, or
  when the best head fails to beat the uniform baseline
  (`lcs_lift_canonical <= 0`) — a mechanically degenerate result the jury can
  skip.
