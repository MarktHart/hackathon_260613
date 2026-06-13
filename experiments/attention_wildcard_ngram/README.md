# attention_wildcard_ngram

## Question
Can an attention head implement **wildcard n-gram matching** — attending from
a target token back to an anchor token while "skipping over" intervening
*wildcard* positions that may hold any token? In the pattern `A * B` (where
`*` is a wildcard), does `B` attend sharply to `A` regardless of what occupies
the wildcard slot, and how far does that hold as the gap widens to `A * * B`,
`A * * * B`, …?

## Setup
**Synthetic generator only.** No trained model is used. The task generates
sequences of the form:
```
[anchor, wildcard_1, ..., wildcard_k, target, filler...]
```
where:
- `anchor` is a fixed token (ID 1) at position 0;
- `wildcard_i` are drawn uniformly from a distractor vocabulary (IDs 10..31);
- `target` is a fixed token (ID 2) that should attend back to `anchor`;
- `filler` is the padding token (ID 0) padding to a fixed length.

The **wildcard span** `k` (number of wildcard tokens between anchor and target)
is the experimental knob. We test `k ∈ {0, 1, 2, 3, 4}`. With `k = 0`, anchor
and target are adjacent — the plain n-gram condition.

**Canonical measurement condition**: `wildcard_span = 1` (one wildcard between
anchor and target), 1024 sequences, sequence length 16, vocabulary size 32.

## Model Function Signature
Attempts must provide a `model_fn` with this exact signature:
```python
def model_fn(batch: Batch) -> np.ndarray:
    """
    Args:
        batch: Batch with `sequences` (n_sequences, seq_len) of token IDs and
               position metadata (anchor_pos, wildcard_pos, target_pos, ...).
    Returns:
        attn: float array of shape (n_sequences, seq_len, seq_len) giving
              attention weights from each query position to each key position.
              Only the target row (query = target_pos) is read by evaluate().
    """
```
The function computes attention weights for the *single head* the attempt
claims implements wildcard skipping. `evaluate` indexes `[:, target_pos, :]`
to obtain the target's attention distribution over keys. The smoke-test
reference (`random_model_fn`) returns uniform attention `1/seq_len`.

## Payload Contract
`task.evaluate(model_fn)` returns a dict with these keys:

```python
{
    "version": 1,                      # payload schema version (== benchmark.VERSION)
    "canonical_span": 1,               # wildcard_span designated canonical
    "seq_len": 16,                     # sequence length used
    "vocab_size": 32,                  # vocabulary size used
    "anchor_token": 1,                 # token ID for anchor
    "target_token": 2,                 # token ID for target
    "wildcard_token_range": [10, 31],  # inclusive range for wildcard tokens
    "sweep": [                         # exactly 5 records, spans 0..4 ascending
        {
            "wildcard_span": int,            # k, number of wildcard tokens (0..4)
            "wildcard_pos": int,             # position of first wildcard (always 1)
            "target_pos": int,               # position of target token (1 + k)
            "n_sequences": int,              # always 1024
            "mean_attn_on_anchor": float,    # mean target→anchor weight
            "mean_attn_on_wildcards": float, # mean target→wildcard weight (0 if k=0)
            "mean_attn_on_others": float,    # mean target→other-position weight
            "sharpness": float,              # anchor / (wildcards + others + 1e-8)
        },
        ...
    ],
}
```
All floats are Python `float`. The sweep always contains exactly 5 records
(spans 0–4) in ascending order. `benchmark.score` requires only
`wildcard_span` and `sharpness` per record; the other fields are
self-describing context.

## Metrics
`benchmark.score(payload)` returns a flat dict (`version` first):

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `wildcard_skip_robustness` | `sharpness_span_1 / sharpness_span_0` | **bigger is better** | Headline. How well sharpness survives inserting one wildcard vs. none. ~1.0 = clean skip; <1 = degradation; clamped to 0 if span-0 sharpness ≤ 0. |
| `sharpness_canonical` | `sharpness` at `wildcard_span=1` | bigger is better | Sharpness at the canonical condition. |
| `mean_sharpness` | mean of `sharpness` over spans 0–4 | bigger is better | Sharpness averaged across the sweep. |
| `sharpness_wildcard_span_K` | `sharpness` at span `K` | bigger is better | Per-slice sharpness, `K ∈ {0,1,2,3,4}`. |
| `linear_baseline_sharpness_wildcard_span_K` | uniform-attention sharpness at span `K` | — | Neutral reference: `(1/L) / (mean_wild + 1/L + 1e-8)`, with `mean_wild = 1/L` for `K>0` else `0`. ≈ 0.5 for `K>0`, ≈ 1.0 for `K=0` (with `L=16`). |
| `lift_over_baseline_canonical` | `sharpness_canonical − linear_baseline_sharpness_wildcard_span_1` | bigger is better | Absolute improvement over uniform attention at the canonical span. |

**Reading the sharpness scale.** A head that puts essentially all of the
target's attention on the anchor drives `mean_attn_on_wildcards` and
`mean_attn_on_others` toward 0, so `sharpness` grows large (the `1e-8` keeps it
finite). Uniform attention yields ~0.5 for spanned conditions; a real wildcard
matcher should be many times higher.

## Bump Procedure
Bump `VERSION` in `benchmark.py` and `version` in the payload **together** when:
- any metric formula changes;
- a payload key is added, removed, or retyped;
- the canonical span changes;
- the sweep spans change (e.g. adding span 5).

Adding a new metric without touching existing ones does **not** require a bump.
After bumping, update the "Payload Contract" and "Metrics" tables here in the
same commit.

## Optional Pipeline Hooks
- `GPU_REQUIREMENT = 1` (default; attempts run on the GPU).
- `is_obviously_broken(metrics)` returns `True` if any metric is NaN/inf, if
  `wildcard_skip_robustness < 0.5`, or if `sharpness_canonical ≤
  linear_baseline_sharpness_wildcard_span_1`. This only short-circuits the
  jury for clearly degenerate attempts; it never adjudicates borderline ones.
