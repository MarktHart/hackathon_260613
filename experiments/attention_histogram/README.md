# attention_histogram

## The question

When an attention head must single out **one** correct key among several
distractor keys, what does the *histogram* of its attention weights look like —
and does it stay sharp and correctly aimed as the distractors become more
similar to the true target?

A good attention mechanism produces a **single-peaked, low-entropy** attention
distribution concentrated on the right key. A weak one smears its mass across
positions (high entropy) or peaks on the wrong key. We sweep the
distractor↔target cosine similarity to push interference up and watch the
histogram degrade.

## Setup

**Fully synthetic.** No trained model, no dataset, no GPU needed for the
data/metric (attempts still run on GPU per the framework — see
`GPU_REQUIREMENT`).

For each `(similarity, seed)` condition the generator builds:

- a target key direction `t` on the unit sphere in `R^d`;
- `n_positions` key vectors: one is the target `t`; each distractor has cosine
  `≈ similarity` to `t` (so larger `similarity` ⇒ harder);
- a query `q = unit(t + noise·ε)` — it points at the target but is corrupted;
- the index `target_index` of the correct key.

The attempt's mechanism scores every key position for the query. We softmax
those scores into an attention distribution and measure its shape and its peak.

## Canonical measurement condition

| parameter            | value                         |
|----------------------|-------------------------------|
| `d`                  | 32                            |
| `n_positions`        | 16                            |
| `key_sim_sweep`      | `[0.0, 0.2, 0.4, 0.6, 0.8]`   |
| `canonical_similarity` | `0.0` (distinct keys)       |
| `n_seeds`            | 16 per sweep point            |
| query noise          | 0.6                           |
| `EVAL_SEED`          | 7                             |

`generate(seed)` is deterministic: same seed → same batch. `evaluate` always
uses `EVAL_SEED`.

## The `model_fn` contract

Attempts provide a single callable:

```python
def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    query : (d,)            float32  — the query vector
    keys  : (n_positions,d) float32  — candidate key vectors
    returns: (n_positions,) float    — attention LOGITS (pre-softmax)
    """
```

`evaluate` applies the softmax itself, so attempts only return raw scores. The
signature is intentionally narrow — one query, one key matrix, one logit
vector. `task.random_model_fn()` returns such a callable emitting random logits
(used by the smoke test).

## Payload contract

`task.evaluate(model_fn)` returns exactly:

| key                    | type              | semantics                                       |
|------------------------|-------------------|-------------------------------------------------|
| `version`              | int (== 1)        | payload schema version                          |
| `model_name`           | str               | self-describing label                           |
| `d`                    | int               | key dimensionality                              |
| `n_positions`          | int               | number of key positions per query              |
| `chance_hit_rate`      | float             | `1/n_positions`                                 |
| `canonical_similarity` | float             | the default condition on the sweep axis         |
| `key_sim_sweep`        | list[float]       | the sweep axis values                           |
| `sweep`                | list[record]      | one record per sweep value (attempt)            |
| `linear_baseline`      | list[record]      | one record per sweep value (dot-product ref)    |

Each `sweep` record:

| field                | type  | meaning                                              |
|----------------------|-------|------------------------------------------------------|
| `similarity`         | float | nominal distractor cosine                            |
| `attention_sharpness`| float | `1 − H(attn)/log(n)`, in `[0,1]` (uniform 0, peak 1) |
| `attention_entropy`  | float | Shannon entropy of the histogram, nats               |
| `target_hit_rate`    | float | fraction where `argmax(attn) == target_index`        |
| `n_seeds`            | int   | seeds averaged                                       |

Each `linear_baseline` record: `{similarity, attention_sharpness,
target_hit_rate, n_seeds}` — the **same measurements** computed from plain
dot-product attention `softmax(keys @ query)`, the no-mechanism reference.

## Metrics (`benchmark.score`)

`version` is always the first key. Bigger is better for every metric except
`attention_entropy_*` (smaller = sharper).

| metric                                | meaning                                                              |
|---------------------------------------|----------------------------------------------------------------------|
| `histogram_robustness` *(headline)*   | `sharpness(hardest sim) / sharpness(easiest sim)`, clipped `[0,1]`   |
| `attention_sharpness_canonical`       | sharpness at `canonical_similarity`                                  |
| `target_hit_rate_canonical`           | targeting accuracy at `canonical_similarity`                         |
| `mean_target_hit_rate`                | hit rate averaged over the whole sweep                               |
| `lift_over_baseline_canonical`        | method − dot-product sharpness at canonical                          |
| `attention_sharpness_sim_<v>`         | per-slice sharpness (`0p2` form)                                     |
| `attention_entropy_sim_<v>`           | per-slice histogram entropy (nats)                                   |
| `target_hit_rate_sim_<v>`             | per-slice targeting accuracy                                         |
| `linear_baseline_sharpness_sim_<v>`   | dot-product reference sharpness, same condition                      |
| `linear_baseline_hit_rate_sim_<v>`    | dot-product reference hit rate, same condition                       |

Per-slice floats use `0p2` form (`sim:.1f` with `.`→`p`).

The **headline** `histogram_robustness` rewards a mechanism that keeps its
attention concentrated even when distractors crowd the target. Reading it
alongside `mean_target_hit_rate` separates "sharp but wrong" from "sharp and
right". Compare against the `linear_baseline_*` slices to know whether the
mechanism beats plain dot-product attention.

## Pipeline hooks

- `GPU_REQUIREMENT = 1`.
- `is_obviously_broken(metrics)` skips the jury when results are mechanically
  degenerate: any NaN/inf, near-zero canonical sharpness (uniform attention),
  or canonical hit rate at/below chance (`1/16`). It never fires on a
  borderline-but-real result.

## Bumping `VERSION`

Bump `benchmark.VERSION` (and `task` payload `version`) and update this file in
the same commit when you: change any existing metric formula, rename/remove/
retype a payload key, or change the canonical condition. Adding a new metric, an
optional payload key, or a new sweep slice does not require a bump. Currently
`VERSION = 1`.
