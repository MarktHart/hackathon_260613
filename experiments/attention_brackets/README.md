# attention_brackets

## Question

Does an attention head route a **closing** bracket's query to its **matching
opening** bracket — the opener a parser's stack would pop — rather than to a
cheaper positional heuristic (nearest opener, previous token, uniform)? And how
does that stack-matching behaviour hold up as the **nesting depth** grows?

This is the attention analogue of "the model implements a stack." A head that
truly matches brackets must, at a closing position, look back across arbitrary
intervening (balanced) content to the one opener that pairs with it.

## Setup

**Fully synthetic.** `task.generate(seed)` deterministically produces balanced
single-type bracket sequences (`(` = token `0`, `)` = token `1`, `PAD` = `2`),
grouped by a maximum-nesting-depth bound. For each position we record the index
of its true matching opener using a stack — exactly the parser ground truth.

No trained model is required. Any attempt supplies a `model_fn` that emits an
attention matrix per sequence; the attempt may compute that matrix however it
likes (trained head, hand-built circuit, heuristic).

## Canonical measurement condition

- Sequences are generated with `seed=0` (fixed inside `evaluate`).
- `SEQ_LEN = 24`, `N_PER_DEPTH = 64`, one bracket type.
- Depth sweep: `DEPTHS = (1, 2, 3, 4, 5)`; **canonical depth = 3**.
- Attention is treated as **causal**: query `q` may attend only to keys
  `k <= q`. The evaluator renormalises each row to sum to 1 and clamps
  negatives to 0, so attempts need not pre-mask.

## `model_fn` contract

```python
model_fn(tokens: np.ndarray[int32, (L,)]) -> np.ndarray[float, (L, L)]
```

- Input: token ids of **one** sequence, length `L = SEQ_LEN`.
- Output: a real matrix `A` where `A[q, k]` is the weight query `q` places on
  key `k`. Must be finite. It need not be pre-normalised or pre-masked —
  `evaluate` applies a causal mask, clips negatives, and renormalises rows.

`task.random_model_fn()` returns a strawman with this exact signature emitting
random attention; it should land near the uniform baseline.

## Payload contract (`task.evaluate` → `benchmark.score`)

```python
{
  "version": 1,
  "config": {
    "depths": [1, 2, 3, 4, 5],
    "canonical_depth": 3,
    "n_per_depth": 64,
    "seq_len": 24,
    "bracket_types": 1,
  },
  "sweep": [            # one record per depth, in DEPTHS order
    {
      "depth": int,                  # nesting-depth bound for this slice
      "n_closers": int,              # number of closing brackets scored
      "match_accuracy": float,       # frac. where argmax(A[q]) == matching opener
      "match_mass": float,           # mean A[q, matching_opener]
      "uniform_baseline_mass": float # mean 1/(q+1): uniform causal reference
    },
    ...
  ],
}
```

`score()` raises `ValueError`/`KeyError` if the sweep length, depth ordering,
keys, or numeric finiteness are violated.

## Metrics

All in `[0, 1]`, **bigger is better**.

| metric | meaning |
|---|---|
| `bracket_match_robustness` | **headline.** Min normalised lift over uniform across all depths — how well matching survives the deepest nesting. |
| `bracket_match_accuracy_canonical` | argmax-matching accuracy at depth 3. |
| `bracket_match_mass_canonical` | mean attention mass on the matching opener at depth 3. |
| `uniform_baseline_mass_canonical` | uniform causal reference mass at depth 3. |
| `lift_over_uniform_canonical` | `(mass − baseline)/(1 − baseline)` at depth 3. |
| `match_accuracy_depth_<d>` | per-slice argmax accuracy. |
| `match_mass_depth_<d>` | per-slice matching mass. |
| `uniform_baseline_mass_depth_<d>` | per-slice uniform reference. |
| `match_lift_depth_<d>` | per-slice normalised lift over uniform. |

Normalised lift: `lift = (match_mass − baseline) / (1 − baseline)`, clamped to
`[0, 1]`; `0` when there is no headroom. A method *beating the uniform
baseline* is the meaningful signal — the raw mass alone is not.

## Pipeline hooks

- `GPU_REQUIREMENT = 0` — synthetic NumPy, no accelerator.
- `is_obviously_broken` — `True` on any NaN/inf, or when canonical matching mass
  fails to clear the uniform floor by 25% (a head with no matching mechanism).

## Bump procedure

Bump `VERSION` (in both `task.py`'s payload and `benchmark.py`) when you change
any metric formula, rename/retype a payload key, or change the canonical
condition (canonical depth, sweep, seq len, normalisation). Adding a new metric
or a new depth slice does **not** require a bump. Update this contract in the
same commit.
