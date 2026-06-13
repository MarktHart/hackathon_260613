# Attention Edit Distance

## Question

Do attention patterns in a language model change smoothly with the edit distance between input sequences? Specifically, if we take two sequences with a known Levenshtein edit distance, does the distance between their attention patterns (at a specific layer/head) correlate monotonically with that edit distance?

This tests whether attention mechanisms track syntactic structure in a way that respects edit operations — a prerequisite for any mechanistic claim about "the model computes edit distance in attention."

## Setup

**Synthetic generator.** We generate pairs of token sequences where:
- Base sequence: random tokens from a vocabulary of size 100, length `L = 32`
- Edited sequence: base sequence with `k` random edit operations (insert, delete, substitute)
- Edit distance `k` ranges from 0 to 8 in steps of 1

**Model.** A fixed pre-trained model (canonical: GPT-2 small, layer 5, head 3 — chosen as a mid-layer attention head that attends to syntactic structure). The model is frozen; attempts only provide a `model_fn` that extracts attention weights.

**Canonical measurement condition:**
- Model: `gpt2` (124M params)
- Layer: 5 (0-indexed)
- Head: 3 (0-indexed)
- Sequence length: 32
- Vocabulary: first 100 tokens of GPT-2 tokenizer
- Edit distances: 0, 1, 2, 3, 4, 5, 6, 7, 8
- 50 sequence pairs generated per target edit count `k` (450 pairs total); pairs are then bucketed by their *measured* Levenshtein distance, so per-bucket `n_pairs` may differ slightly from 50 (a sequence of `k` edits can yield a true distance `< k` when edits cancel or substitute a token to itself). Each sweep entry reports its actual `n_pairs`.
- Attention distance metric: 1 - cosine similarity between flattened attention matrices [seq_len, seq_len]

## Payload Contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                                    # payload schema version
    "model_name": "gpt2",                            # canonical model identifier
    "layer": 5,                                      # layer index
    "head": 3,                                       # head index
    "seq_len": 32,                                   # sequence length
    "vocab_size": 100,                               # vocabulary size used
    "attention_distance_metric": "1_minus_cosine",   # how attn distance is computed
    "sweep": [
        {
            "edit_distance": int,                    # true Levenshtein distance (0-8)
            "attn_distance_mean": float,             # mean attention distance over pairs
            "attn_distance_std": float,              # std dev across pairs
            "n_pairs": int,                          # number of pairs at this edit distance
        },
        ... (one per edit distance value)
    ],
    "linear_baseline": {
        "attn_distance_mean": list[float],           # same length as sweep, baseline means
        "attn_distance_std": list[float],            # baseline stds
    }
}
```

- `model_fn(tokens: np.ndarray) -> np.ndarray` signature:
  - `tokens`: `[batch, seq_len]` int32, token IDs
  - Returns: `[batch, seq_len, seq_len]` float32, attention weights for the **canonical layer/head only** (batch x query x key)
  - The function must handle variable batch sizes. No gradients, no training mode.

## Metrics

`benchmark.score(payload)` returns a flat dict:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `version` | `payload["version"]` | — | Schema version |
| `edit_distance_correlation` | Spearman ρ between `edit_distance` and `attn_distance_mean` across sweep | Bigger = better | Headline summary: monotonic relationship strength |
| `edit_distance_correlation_pearson` | Pearson r between `edit_distance` and `attn_distance_mean` | Bigger = better | Linear relationship strength |
| `attn_distance_edit_<k>` | `attn_distance_mean` at edit distance `k` | — | Per-slice values for diagnostic panel |
| `linear_baseline_correlation` | Spearman ρ for `linear_baseline.attn_distance_mean` vs edit distance | Smaller = better (negative is ideal) | Random-attention baseline |
| `lift_over_baseline` | `edit_distance_correlation - linear_baseline_correlation` | Bigger = better | Improvement over random attention |

**Edge cases:**
- If `n_pairs == 0` for any sweep entry, that entry is skipped in correlation.
- If all `attn_distance_mean` are identical (zero variance), correlation = 0.
- `linear_baseline` is computed once per `benchmark.VERSION` using a fixed seed and stored in the payload by `task.evaluate`.

## Bump Procedure

`benchmark.VERSION` increments when:
- The correlation formula changes (Spearman → Pearson, etc.)
- Payload keys are added/removed/retyped
- Canonical model/layer/head/seq_len/vocab changes
- `attention_distance_metric` changes

Adding a new per-slice metric (e.g., `attn_distance_std_edit_<k>`) does **not** require a version bump.