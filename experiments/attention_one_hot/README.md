# Attention One-Hot Goal

## Question
Can an attention head implement a **one-hot attention pattern** — attending to exactly one target position while ignoring all others — when the query matches a specific key pattern?

This tests the mechanistic capability of attention to perform exact key-value lookup, a primitive for induction heads, copying, and sparse retrieval.

---

## Setup
**Fully synthetic.** No trained models, no external datasets.

- **Embedding dimension**: `d_model = 32`
- **Sequence lengths**: Sweep over `L ∈ {16, 32, 64, 128, 256}`
- **Temperature**: Fixed at `τ = 0.1` (sharp attention)
- **Key construction**: One "target" key at a random position matches the query; all other keys are orthogonal noise.
- **Values**: a distinct random unit vector per position (an internal value codebook used only to compute `output_cosine`; attempts never see it). Per-position directions make `output_cosine` reflect *where* attention landed.

The canonical measurement condition is `L = 64`.

---

## Canonical Measurement Condition
Every attempt **must** evaluate at `L = 64` with `τ = 0.1`, `d_model = 32`, using the deterministic seed `0` for the target position. This single number is the headline result.

---

## Model Function Signature
```python
def model_fn(query: np.ndarray, keys: np.ndarray, temperature: float) -> np.ndarray:
    """
    Compute the attention weight distribution for a single head.

    Args:
        query:  (d_model,)            — the query vector
        keys:   (seq_len, d_model)    — key matrix for the sequence
        temperature: float            — softmax temperature τ

    Returns:
        attn_weights: (seq_len,) — non-negative attention weights over positions,
                                   summing to 1. This *is* the attention pattern.
    """
```

The attempt's `main.py` implements this callable. `task.evaluate` calls it once
per sequence length in the sweep, and **all metrics are derived from the returned
weights** — so different attempts produce different scores. `evaluate` combines
the weights with an internal value codebook (not passed to `model_fn`) only to
compute `output_cosine`. Returning the attention *pattern* (not a pre-reduced
output vector) is what lets the benchmark observe and rank each attempt.

---

## Payload Contract
`task.evaluate` returns a dict with exactly these keys:

```python
{
    "version": 1,                              # payload schema version
    "canonical_length": 64,                    # the canonical L
    "temperature": 0.1,                        # fixed τ
    "d_model": 32,                             # fixed embedding dim
    "sweep": [                                 # one record per sequence length
        {
            "length": int,                     # sequence length L
            "target_pos": int,                 # 0-indexed position of the target key
            "attention_entropy": float,        # entropy of the attention distribution (nats)
            "peak_attention": float,           # max attention weight (should be ≈1 for one-hot)
            "target_attention": float,         # attention weight on the true target position
            "output_cosine": float,            # cosine between attention output (weights·codebook) and the target value vector
        },
        ...
    ]
}
```

All floats are plain Python `float`. The sweep order matches the canonical length order `[16, 32, 64, 128, 256]`.

---

## Metrics
Computed by `benchmark.score(payload)`.

| Metric | Formula | Direction | Notes |
|--------|---------|-----------|-------|
| `one_hot_canonical` | `target_attention` at `L=64` | **bigger is better** | Headline summary. 1.0 = perfect one-hot. |
| `one_hot_length_<L>` | `target_attention` at each `L` | **bigger is better** | Per-slice. Keys use `length_16`, `length_32`, etc. |
| `peak_attention_length_<L>` | `peak_attention` at each `L` | **bigger is better** | Should track `target_attention` if the peak is on target. |
| `entropy_length_<L>` | `attention_entropy` at each `L` | **smaller is better** | 0 = perfect one-hot. |
| `output_cosine_length_<L>` | `output_cosine` at each `L` | **bigger is better** | End-to-end: alignment of the attention output with the target's value vector. |
| `linear_baseline_one_hot_length_<L>` | `1 / L` | **reference** | Uniform attention baseline. |
| `length_robustness` | `min_L(target_attention) / max_L(target_attention)` | **bigger is better** | Ratio in `[0,1]`; measures degradation with scale. |
| `version` | `benchmark.VERSION` | — | Always first key. |

---

## Bump Procedure
- `VERSION` in `benchmark.py` increments on any incompatible payload/metric change.
- Update this README's **Payload Contract** and **Metrics** tables in the same commit.
- Old `benchmark.json` files remain on disk; the dashboard filters to the highest version.