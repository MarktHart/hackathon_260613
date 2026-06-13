# attention_cyk

## Question

Can an attention mechanism implement the **CYK inside recursion** — for a
given chart cell, does it place its attention mass on the *split point* that
actually combines two sub-spans into the cell?

CYK parses a string under a Chomsky-Normal-Form grammar by filling a chart:
`chart[i][j]` is the set of nonterminals that derive `seq[i:j]`. A cell of
span `≥ 2` is filled exactly when there is a split point `k` (`i < k < j`) and
a binary production `P -> L R` with `L ∈ chart[i][k]` and `R ∈ chart[k][j]`.
The split point is the single piece of structure a soft-attention head would
have to discover to run the algorithm. This goal scores whether a mechanism
finds it.

## Setup

Fully **synthetic**. A fixed CNF grammar (`dyck1_v1`) for the **Dyck-1
language** of balanced brackets, over two terminals (`(` = 0, `)` = 1) and
four nonterminals (`S, X, L, R`):

```
S -> L X | L R | S S        X -> S R        L -> '('     R -> ')'
```

A span `(i, j)` is labelled `S` iff `seq[i:j]` is a balanced bracket string.
Its correct split points are exactly the **balance points** — where a prefix
returns to bracket-depth 0 (an `S S` split) or where the opening bracket
matches (an `L X` split). This keeps the chart **sparse**: most cells are
empty, and a filled cell admits only a few of its candidate splits, so uniform
attention scores well below 1.0 and a real mechanism can beat it.

`task.generate` draws random bracket strings of length 4–9, keeps those with
at least one filled span-`≥3` cell, and runs exact CYK to label every cell and
its correct split points. The grammar is fixed, so `generate(seed)` accepts a
seed for determinism; the canonical batch is `seed=0` (50 strings).

## Canonical measurement condition

Every attempt is scored on the canonical batch (`generate(seed=0)`) over
**every query cell** — every `(i, j)` with `j - i ≥ 3` and `chart[i][j]`
non-empty. Span-2 cells are excluded: they have a single forced split point,
so every mechanism scores them 1.0 and they carry no signal. There is no
condition to choose; the headline metric is the cell-weighted accuracy over
that whole set.

## model_fn contract

```python
model_fn(seq: tuple[int, ...], i: int, j: int) -> np.ndarray
```

- `seq` — the input string as a tuple of terminal ids (`(` = 0, `)` = 1).
- `i`, `j` — the half-open span of the chart cell being queried
  (`0 ≤ i < j ≤ len(seq)`, `j - i ≥ 3`).
- **returns** a 1-D NumPy array of shape `(len(seq) + 1,)` of **nonnegative**
  scores over split positions. The evaluator reads positions `i+1 … j-1` (the
  valid splits), clips negatives to 0, normalises them to a probability
  distribution, and measures the mass on correct splits. If the valid scores
  sum to 0 the cell is treated as uniform attention.

The mechanism is queried once per cell. `task.random_model_fn()` returns a
model_fn of this exact signature that emits random scores (≈ uniform baseline)
and is used by the smoke test.

## Payload contract

`task.evaluate(model_fn)` returns:

| key           | type         | semantics |
|---------------|--------------|-----------|
| `version`     | `int`        | payload schema version (currently `1`) |
| `grammar`     | `str`        | grammar id (`"dyck1_v1"`) |
| `num_strings` | `int`        | strings in the canonical batch |
| `max_len`     | `int`        | max string length used |
| `sweep`       | `list[dict]` | one record per span length present |

Each `sweep` record:

| key                | type    | semantics |
|--------------------|---------|-----------|
| `span_len`         | `int`   | cell span `j - i` for this slice |
| `num_cells`        | `int`   | query cells of this span |
| `split_accuracy`   | `float` | mean probability mass on correct splits, in `[0, 1]` |
| `uniform_baseline` | `float` | mean `#correct / #valid` over those cells, in `[0, 1]` |

## Metrics

`benchmark.score(payload)` returns (bigger is better unless noted):

| metric | meaning |
|--------|---------|
| `version` | benchmark version (filter key) |
| `cyk_split_accuracy_canonical` | **headline** — cell-weighted mass on correct splits over all query cells |
| `uniform_baseline_accuracy` | same quantity under uniform attention |
| `lift_over_uniform` | `canonical − baseline`; the mechanism's signal above chance |
| `split_accuracy_len_<L>` | per-slice accuracy for span length `L` |
| `uniform_baseline_len_<L>` | per-slice uniform baseline for span length `L` |
| `split_accuracy_robustness` | `min_slice_acc / max_slice_acc` in `[0, 1]`; stability across span length |
| `num_query_cells` | total query cells scored (int) |

A meaningful result has `cyk_split_accuracy_canonical` well above
`uniform_baseline_accuracy` (positive `lift_over_uniform`), held across span
lengths (`split_accuracy_robustness` near 1).

`is_obviously_broken` flags NaN/inf metrics or a canonical accuracy strictly
below the uniform baseline (worse than chance), which short-circuits the jury.
`GPU_REQUIREMENT = 1`.

## Bump procedure

Bump `VERSION` (in `benchmark.py`) and update this contract together when you:
change any metric formula; rename/remove/retype a payload key; or change the
grammar or canonical batch. Adding a new metric or a new span-length slice
does not require a bump. Old `benchmark.json` files stay on disk; the
dashboard filters to the highest version.
