# Attention Substring: In-Context Substring Matching

## Question
**When a token sequence contains a repeated multi-token pattern, does an
attention head at the *second* occurrence's continuation point attend back to
the *first* occurrence — i.e. does the model implement substring (induction-
style) matching, and how reliably across pattern length and inter-occurrence
distance?**

Given a sequence `... [P] ... [P] X ...` where `[P]` is a short pattern that
appears twice, an induction-style head sitting at the position right after the
second `[P]` should attend back to the matching position inside the *first*
`[P]`. This goal measures how often, and how sharply, the model's best head
does exactly that.

---

## Setup
**Synthetic generator (`task.generate`).** We build sequences with a planted,
twice-repeated pattern:

```
[BOS] [prefix] [pattern] [filler] [pattern] [target] [suffix]
```

- `seq_len = 64`, `vocab_size = 64`
- Reserved tokens: `BOS = 0`, `PAD = 1`; content tokens drawn from `2..63`
- Pattern lengths swept: `[2, 3, 4]`
- Inter-occurrence distances (filler length) swept: `[8, 16, 32]`
- `n_per_combo = 50` sequences per (pattern_length, distance) cell
- Total sequences: `3 × 3 × 50 = 450`

For each sequence the first pattern starts at position `1` (right after `BOS`).
The second pattern starts at `1 + plen + dist`, and the `target` token is placed
immediately after it at `target_pos = 1 + 2*plen + dist`. The **source
position** — the position the matching head should attend to — is the last
token of the *first* pattern, `source_pos = plen`. The `target` token itself is
a deterministic function of the pattern (`(pattern.sum() + plen*7) % 62 + 2`)
so an optional next-token head can be scored too. Filler, prefix and suffix are
random content tokens.

`generate` is **deterministic** for a given seed (`numpy.random.RandomState`).
The canonical seed is **`42`**.

---

## Canonical Measurement Condition
Every attempt **must** evaluate on `generate(seed=42)` with the canonical
hyperparameters above (this is what `task.evaluate` uses internally — attempts
do not choose the seed). The full sweep of 450 sequences is fixed.

---

## Model Function Signature
```python
def model_fn(input_ids: np.ndarray) -> dict:
    """
    Args:
        input_ids: np.ndarray of shape [1, seq_len] (int32). evaluate() calls
            the model one sequence at a time (batch dimension of 1).

    Returns:
        dict with:
          - "attn_weights": np.ndarray of shape
                [n_layers, n_heads, seq_len, seq_len] (float32), the per-head
                attention matrices (no batch dim). For each (layer, head, query),
                the key dimension is expected to be a valid attention
                distribution.
          - "logits": optional np.ndarray of shape [seq_len, vocab_size]
                (float32). If present, enables next-token accuracy scoring; if
                omitted, that metric is simply not emitted.
    """
```
`n_layers` and `n_heads` are read from the returned `attn_weights` shape — the
model declares its own layer/head count. Attempts wrap their model (PyTorch,
JAX, etc.) to match this signature. The framework calls
`task.evaluate(model_fn)`.

---

## Payload Contract
`task.evaluate` returns a dict with exactly these keys:

| Key | Type | Description |
|-----|------|-------------|
| `version` | int | Payload schema version (always `1` for this goal). |
| `config` | dict | Fixed hyperparameters: `seq_len`, `vocab_size`, `pattern_lengths`, `distances`, `n_per_combo`, `seed`. Self-describing; provenance for the run. |
| `sweep` | list[dict] | One record **per generated sequence** (450 of them). Each record captures the *best head*'s behaviour at the target position (see below). |

For each sequence, `evaluate` selects the **best head** as the
`(layer, head)` with the largest attention weight on `source_pos`, and records
that head's behaviour. Each `sweep` record has exactly these keys:

| Key | Type | Description |
|-----|------|-------------|
| `pattern_length` | int | Pattern length for this sequence (`2`, `3`, or `4`). |
| `distance` | int | Inter-occurrence distance / filler length (`8`, `16`, or `32`). |
| `seq_idx` | int | Index of the sequence within its cell (`0..49`). |
| `target_pos` | int | Query position the head is read at. |
| `source_pos` | int | The "correct" key position (last token of first pattern). |
| `attn_to_source` | float | Best head's attention weight from `target_pos` to `source_pos`. |
| `max_attn_elsewhere` | float | Best head's max attention weight to any position **other than** `source_pos` and `target_pos`. |
| `correct_top1` | bool | Whether the best head's argmax attention position equals `source_pos`. |
| `target_token` | int | The planted target token. |
| `predicted_token` | int | `argmax(logits[target_pos])` if logits were returned, else `-1`. |

Example record:
```json
{"pattern_length": 2, "distance": 8, "seq_idx": 0, "target_pos": 13,
 "source_pos": 2, "attn_to_source": 0.81, "max_attn_elsewhere": 0.04,
 "correct_top1": true, "target_token": 37, "predicted_token": 37}
```

---

## Metrics
`benchmark.score(payload)` returns a flat dict of named scalars:

| Metric | Formula | Better |
|--------|---------|--------|
| `version` | `VERSION` (always `1`). | — |
| `substring_detection_canonical` | mean of `correct_top1` over all 450 records. **Headline.** | **Higher** |
| `substring_detection_plen_{L}_dist_{D}` | mean `correct_top1` within the `(pattern_length=L, distance=D)` cell, for `L∈{2,3,4}`, `D∈{8,16,32}` (9 cells). | **Higher** |
| `substring_detection_plen_{L}` | marginal mean `correct_top1` by pattern length. | **Higher** |
| `substring_detection_dist_{D}` | marginal mean `correct_top1` by distance. | **Higher** |
| `attn_ratio_canonical` | mean of `clip(attn_to_source / max_attn_elsewhere, 0, 100)` over all records (when the denominator is ~0, the ratio is `100` if `attn_to_source>0` else `0`). | **Higher** |
| `token_prediction_accuracy` | mean of `predicted_token == target_token` over records with a valid (`≥0`) `predicted_token`. **Only emitted when logits were provided.** | **Higher** |
| `random_baseline_detection` | `1 / (seq_len - 1)` ≈ `0.0159` — chance that a uniformly random argmax position equals `source_pos`. | — |
| `lift_over_random` | `substring_detection_canonical - random_baseline_detection`. | **Higher** |

**Headline metric:** `substring_detection_canonical` — how often the model's
best head correctly points the target position back at the source position.

**Per-slice metrics:** the 9 `substring_detection_plen_{L}_dist_{D}` cells plus
the marginal `_plen_{L}` and `_dist_{D}` breakdowns, so a grader can see where
matching holds (short patterns / short distances) and where it breaks.

**Baseline:** `random_baseline_detection` (uniform-attention chance) with
`lift_over_random` as the method-minus-baseline contrast. A model is only
meaningfully matching if detection sits well above this baseline.

### Edge cases
- Empty sweep / empty slices: `score` raises on a wholly empty `sweep`; any
  empty per-slice cell yields `0.0` (no `ZeroDivisionError`).
- Zero denominator in `attn_ratio`: handled by the clip rule above (no `inf`).
- `seq_len ≤ 1`: `score` falls back to `seq_len = 64` for the baseline.
- `is_obviously_broken(metrics)` short-circuits the jury on NaN/inf, on
  `substring_detection_canonical ≤ 1.5 ×` baseline, or on a provided
  `token_prediction_accuracy < 0.01`.

---

## Bump Procedure
- **Bump `VERSION`** (in `benchmark.py`) and update this README's contract in
  the same commit when:
  - the detection / ratio / baseline formula changes;
  - any `sweep` record key or `config` key is added-as-required, removed, or
    retyped;
  - the canonical condition changes (seed, `seq_len`, `vocab_size`, the
    `pattern_lengths`/`distances` sweep, or `n_per_combo`).
- **No bump needed** for:
  - adding a new derived metric that doesn't change existing ones;
  - adding an optional payload key with a default (e.g. richer per-head debug
    fields) that older consumers can ignore.
- The dashboard filters to the highest `version` present, so old
  `benchmark.json` runs stay legible without polluting the active series.
