# attention_palindrome

## The question

Can a model detect **palindromes** — a property that lives in the *alignment* of
mirrored token positions, not in the *bag of tokens*? Palindrome-ness is the
canonical "you must compare position `i` with position `L-1-i`" task. A genuine
mirror-comparison mechanism (e.g. an attention head that attends `i -> L-1-i`
and checks token equality) should stay sharp even when only a **single** mirror
pair is broken. A histogram / frequency shortcut cannot, because the token
multiset is — by construction — almost uninformative about palindrome-ness.

This goal measures how cleanly an attempt's model separates perfect palindromes
from near-palindromes, as a function of how subtle the corruption is.

## Setup

**Synthetic.** Deterministic generator in `task.py`:

- Sequences of length `SEQ_LEN = 16` over a vocabulary of `VOCAB = 8` tokens,
  giving `HALF = 8` mirror pairs.
- **Positives:** `N_POS = 256` perfect palindromes (random first half, mirrored).
- **Negatives:** for each difficulty `k` in the sweep `MISMATCH_SWEEP = (1, 2, 4, 8)`,
  `N_NEG = 256` sequences that start as a perfect palindrome and have **exactly
  `k` mirror pairs broken** (the right-hand token of each chosen pair reassigned
  to a different value). Breaking a pair perturbs the token histogram in a
  class-*uninformative* direction, so a bag-of-tokens readout sits at chance.

The same positive pool is reused as the positive class for every slice's AUC.

Attempts may produce the model_fn however they like — train a tiny transformer,
hand-wire an attention head, or anything else. The data and metric are fixed
here so two attempts can never disagree on either.

## Canonical measurement condition

`task.generate(seed=0)` maps to the canonical seed (`42`) and builds the full
batch above. `task.evaluate` always uses this canonical batch. The **canonical
anchor** for the headline-adjacent metrics is `k = 1` (a single broken pair) —
the most diagnostic slice, where only a true alignment mechanism survives.

## The model_fn contract

```python
model_fn(batch: Batch) -> np.ndarray   # shape (n_seq,), float
```

- `Batch` (from `task.py`) has `tokens` `(n_seq, SEQ_LEN) int32`,
  `is_palindrome` `(n_seq,) bool`, and `mismatch` `(n_seq,) int32`
  (`0` for positives, `k` for slice-`k` negatives).
- Return a real-valued **palindrome score** per sequence, **higher = more
  palindrome-like**. Absolute scale is irrelevant — only the *ordering* of
  scores within the batch is scored (the benchmark uses rank-based AUC, so any
  monotonic transform gives the same result).

Attempts never build the payload themselves: they hand their `model_fn` to
`task.evaluate(model_fn)`, which returns the ready-to-record payload.

## Payload contract

`task.evaluate` returns a dict consumed verbatim by `benchmark.score`:

| key                 | type                    | semantics |
|---------------------|-------------------------|-----------|
| `version`           | `int`                   | payload schema version (currently `1`) |
| `canonical_seed`    | `int`                   | seed used to build the batch (`42`) |
| `seq_len`           | `int`                   | sequence length (`16`) |
| `vocab_size`        | `int`                   | token alphabet size (`8`) |
| `n_pos`             | `int`                   | number of positive palindromes (`256`) |
| `n_neg_per_slice`   | `int`                   | negatives per difficulty slice (`256`) |
| `mismatch_sweep`    | `list[int]`             | the difficulty axis `[1, 2, 4, 8]` |
| `sweep`             | `list[record]`          | the model's per-slice AUC (see below) |
| `linear_baseline`   | `list[record]`          | the histogram-baseline per-slice AUC |

Each `record` (one per `k`, in sweep order):

| key        | type    | semantics |
|------------|---------|-----------|
| `mismatch` | `int`   | broken-pair count `k` for this slice |
| `auc`      | `float` | rank-AUC of positives vs slice-`k` negatives, in `[0, 1]` |
| `n_pos`    | `int`   | positives used in this AUC |
| `n_neg`    | `int`   | slice-`k` negatives used in this AUC |

`sweep` is the attempt's `model_fn`; `linear_baseline` is a closed-form
ridge-on-token-histogram readout (the no-mechanism reference), measured under
identical conditions. Both are computed inside `task.evaluate`.

## Metrics

All metrics are **bigger-is-better**. AUC is the rank-based probability that a
random positive outscores a random negative; chance is `0.5`. "Skill" rescales
AUC to `[0, 1]` via `skill = max(0, 2·(auc − 0.5))`.

| metric | meaning |
|--------|---------|
| `version` | benchmark version (dashboard filters to the latest) |
| **`palindrome_robustness`** | **headline.** Ratio of hardest-slice skill to easiest-slice skill, in `[0, 1]`. `1.0` ⇒ detection holds even when one pair is broken; `~0` ⇒ only gross corruptions are caught (a shortcut). |
| `palindrome_skill_canonical` | skill at the canonical anchor `k = 1` |
| `auc_canonical` | model AUC at `k = 1` |
| `auc_mismatch_k{1,2,4,8}` | model AUC at each slice |
| `linear_baseline_auc_mismatch_k{1,2,4,8}` | histogram baseline AUC at each slice |
| `linear_baseline_auc_canonical` | histogram baseline AUC at `k = 1` |
| `lift_over_baseline_mismatch_k{1,2,4,8}` | model AUC − baseline AUC per slice |
| `lift_over_baseline_canonical` | model AUC − baseline AUC at `k = 1` |

The histogram baseline should sit near `0.5` everywhere; a real mechanism beats
it at every slice and — crucially — keeps high skill at `k = 1`.

### Pipeline hooks

- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken(metrics)` short-circuits the jury when any metric is
  NaN/inf, or when the model's canonical skill fails to clear the histogram
  baseline's skill by more than `0.05` (i.e. no real mechanism beyond
  bag-of-tokens).

## Smoke test

The pipeline runs:

```python
payload = task.evaluate(task.random_model_fn())
metrics = benchmark.score(payload)
```

`random_model_fn` returns shape-correct random scores (pure NumPy), yielding
AUCs near `0.5` and `palindrome_robustness` near `0` — exercising the full
contract without a real model.

## Bumping `VERSION`

Bump `benchmark.VERSION` (and the payload `version`) when any existing metric
formula changes, when a payload key is renamed/removed/retyped, or when the
canonical condition (seed, sweep, anchor) changes. Adding a new metric or a new
sweep slice that extends the existing record shape does **not** require a bump.
Update this README's contract in the same commit.
