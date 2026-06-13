# attention_not

## The question

Can a single attention head implement a logical **NOT** — an *inhibitory
gate*? Concretely: a query position should attend to the **A-token** when an
"attend" feature `A` is present, **but suppress that attention** when an
inhibitory feature `B` is present. The head must compute "attend to A *unless*
B" — `A AND NOT B`.

The interesting regime is **superposition**: when the attend-feature direction
`e_A` and the suppress-feature direction `e_B` are not orthogonal but share a
cosine `theta`, writing to one direction bleeds into the other. We ask how
robustly the NOT survives as `cos(theta)` grows.

## Setup

Fully synthetic, deterministic. Each condition fixes a sequence of four
positions — `[A-token, B-token, query, answer]` — and a head geometry
(`W_Q, W_K, W_V, W_O`, plus unit feature directions `e_A`, `e_B` at a chosen
angle). Per-sequence binary features `feat_A, feat_B ∈ {0,1}` are drawn
deterministically from the seed. No trained model, no dataset, no GPU.

`task.generate(seed, cos_theta)` is deterministic: same `(seed, cos_theta)`
→ identical `Batch`.

## Canonical measurement condition

Every attempt is evaluated by `task.evaluate(model_fn)` over the **canonical
cos(theta) sweep** `[0.0, 0.2, 0.4, 0.6, 0.8]`. The **canonical anchor** is
`cos = 0.0` (orthogonal features) — the easiest case, against which robustness
is measured. Fixed sizes: `d_model = 64`, `d_head = 16`, `n_seq = 600`,
`seq_len = 4`.

## The model_fn contract

```python
def model_fn(batch: task.Batch) -> dict:
    # returns {"attn_weights": np.ndarray of shape (n_seq, seq_len, seq_len)}
    ...
```

- `attn_weights[s, QUERY_POS, A_POS]` is the attention mass sequence `s`'s
  query position (index `2`) places on the A-token (index `0`).
- Rows should sum to ~1 over the last axis, but the metrics only read the
  query→A entry, so exact normalisation is not enforced.
- Pure NumPy in / NumPy out. The attempt builds the head; `evaluate` calls it
  once per sweep condition and reduces the output to scalars.

`task.random_model_fn()` returns a contract-shaped `model_fn` that emits zeros
of shape `(n_seq, seq_len, seq_len)` — used by the pipeline smoke test.

## Payload contract

`task.evaluate` returns, and `benchmark.score` consumes, exactly:

| key | type | semantics |
|-----|------|-----------|
| `version` | `int` | payload schema version (must equal `benchmark.VERSION`) |
| `config` | `dict` | self-describing run config (sizes, sweep, anchor) — not read by `score` |
| `sweep` | `list[dict]` | one record per cos value, in `CANONICAL_COS` order |
| `baseline` | `list[dict]` | same shape as `sweep`, measured on the no-NOT linear baseline head |

Each `sweep` / `baseline` record:

| key | type | semantics | direction |
|-----|------|-----------|-----------|
| `cos` | `float` | the `cos(theta)` for this slice | — |
| `not_sharpness` | `float` in `[0,1]` | AUC that query→A attention is higher when `B=0` than `B=1` (given `A=1`); ties = 0.5 | bigger better |
| `suppression_gap` | `float` in `[-1,1]` | `mean(attn_to_A | B=0) − mean(attn_to_A | B=1)` given `A=1` | bigger better |
| `attend_specificity` | `float` in `[0,1]` | `1 − mean(attn_to_A | A=0)`; penalises attending to A when A is absent | bigger better |

## Metrics

`benchmark.score(payload) -> dict[str, float | int]`. First key is always
`version`. All metrics are **bigger-is-better**.

| metric | meaning |
|--------|---------|
| `superposition_robustness` | **headline.** Worst-slice NOT-sharpness re-centred on chance and normalised by the canonical slice: `(min_c sharp_c − 0.5) / (sharp_canonical − 0.5)`, clamped to `[0,1]`. `1.0` = no degradation under superposition; `0` = collapses to chance. |
| `not_sharpness_cos_<v>` | per-slice NOT-sharpness (`0p2`-form floats) |
| `linear_baseline_sharpness_cos_<v>` | NOT-sharpness of the no-mechanism baseline, same condition |
| `lift_over_baseline_cos_<v>` | `not_sharpness − baseline` per slice |
| `suppression_gap_cos_<v>` | per-slice suppression gap |
| `attend_specificity_cos_<v>` | per-slice attend specificity |
| `not_sharpness_canonical` | sharpness at the orthogonal anchor |
| `linear_baseline_sharpness_canonical` | baseline sharpness at the anchor |
| `lift_over_baseline_canonical` | lift at the anchor |

The **linear baseline** is a head that attends to A whenever `feat_A=1`,
ignoring `B` entirely — it cannot do NOT, so it sits at chance (`~0.5`). A
method beating it is meaningful; in isolation a sharpness number is not.

### Pipeline hooks

- `GPU_REQUIREMENT` is not exported (defaults to 1; this goal needs no GPU).
- `is_obviously_broken` fires (skips the jury) when any metric is NaN/inf, or
  when canonical NOT-sharpness fails to clear the baseline / chance by `0.1`.

## Bump procedure

Bump `benchmark.VERSION` (currently `1`) and update this contract in the same
commit when you: change any metric formula, rename/retype a payload key, or
change the canonical sweep or anchor. Adding a new metric or an optional,
defaulted payload key does **not** require a bump. Old `benchmark.json` files
stay on disk; the dashboard filters to the highest version present.
