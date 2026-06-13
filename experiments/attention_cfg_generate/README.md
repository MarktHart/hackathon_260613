# attention_cfg_generate

## Question

Do attention heads in a transformer generating strings from a context-free grammar (CFG) exhibit **stack-like attention patterns** — specifically, when predicting a closing parenthesis, does the model attend strongly to its matching opening parenthesis?

This is the generation analogue of the CFG parsing question: during *autoregressive generation* of valid Dyck-1 strings (balanced parentheses), the model must implicitly track the stack of unclosed openings. If attention implements this stack, each `)` token should attend disproportionately to its matching `(` token.

## Setup

- **Synthetic generator only** — no trained model required. The goal provides a deterministic generator of valid Dyck-1 sequences with known parse trees (matching pairs). Attempts supply a `model_fn` that returns attention weights for any input sequence.
- **Grammar**: Dyck-1 (single bracket type `(` `)`), the simplest CFG requiring a stack.
- **Sequences**: Length 32, all valid (properly balanced). Each sequence contains multiple independent well-nested groups, yielding a range of nesting depths 1…D_max.
- **Canonical measurement condition**:
  - Sequence length `L = 32`
  - Vocabulary: `0 = PAD`, `1 = (`, `2 = )`
  - Sweep axis: **nesting depth of the closing parenthesis** (1…5)
  - For each closing token at depth `d`, measure the attention weight it places on its matching opening token, averaged over all heads and all such tokens in the batch.
  - Canonical depth for headline metric: `d = 3` (mid-range, well-sampled).

## Model function signature

```python
def model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    """
    Args:
        input_ids: int32 array, shape [batch, seq_len], values in {0,1,2}

    Returns:
        dict with exactly one key:
        - "attention": float32 array, shape [batch, n_heads, seq_len, seq_len]
          Attention weights *after* softmax, so each [b, h, i, :] sums to 1.
          Rows corresponding to PAD tokens (input_ids == 0) may be arbitrary.
    """
```

Attempts implement this callable. It may wrap a real model, a mechanistic proxy, or a hand-coded stack simulator — the benchmark only consumes the returned attention tensor.

## Payload contract

`task.evaluate(model_fn)` returns a `dict` with exactly these keys:

```python
{
    "version": 1,                              # payload schema version
    "canonical_depth": 3,                      # depth used for headline metric
    "seq_len": 32,                             # sequence length
    "vocab": {"pad": 0, "open": 1, "close": 2},
    "sweep": [                                 # one record per depth d=1..5
        {
            "depth": int,                      # nesting depth of the closing token
            "n_pairs": int,                    # number of (open,close) pairs at this depth in the batch
            "mean_attn_to_match": float,       # mean attention weight from ) to its matching (, averaged over heads and pairs
            "mean_attn_uniform": float,        # 1 / (number of non-PAD tokens in the prefix) — chance baseline
        },
        ...
    ],
    "model_metadata": {                        # free-form, for debugging; ignored by score()
        "n_heads": int,
        "batch_size": int,
    }
}
```

All floats are Python `float` (not numpy scalars). `sweep` is sorted by `depth` ascending.

## Metrics

`benchmark.score(payload)` returns a flat `dict[str, float | int]`:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `version` | `payload["version"]` | — | Schema version (always 1 for this goal) |
| `stack_attention_canonical` | `sweep[d=3]["mean_attn_to_match"]` | **Bigger is better** | Headline: attention on matching `(` at canonical depth 3 |
| `stack_attention_depth_1` … `stack_attention_depth_5` | `sweep[d]["mean_attn_to_match"]` | **Bigger is better** | Per-depth slice for the panel dropdown |
| `uniform_baseline_depth_1` … `uniform_baseline_depth_5` | `sweep[d]["mean_attn_uniform"]` | — | Chance baseline (uniform over prefix) at each depth |
| `lift_over_uniform_canonical` | `stack_attention_canonical - uniform_baseline_depth_3` | **Bigger is better** | Excess over chance at canonical depth |
| `stack_attention_robustness` | `min_d(stack_attention_depth_d) / max_d(stack_attention_depth_d)` | **Bigger is better** | Ratio of worst/best depth; 1 = perfectly depth-invariant |

**Edge cases**: If a depth has `n_pairs == 0`, its slice metrics are `0.0` and excluded from robustness denominator. If all depths have zero pairs, robustness = `0.0`.

## Bump procedure

- `VERSION` in `benchmark.py` and `payload["version"]` in `task.py` move together.
- Bump when: any metric formula changes, a payload key is added/removed/retyped, canonical depth changes, or sweep axis changes.
- Do **not** bump for: adding a new metric, adding an optional payload key with a default, extending `sweep` to deeper depths (the list structure is extensible).
- After bump: update this README's "Payload contract" and "Metrics" tables in the same commit.
</FILE>
