# attention_minimax

## Question

Do attention heads implement **minimax-optimal** attention distributions when the correct target is absent or ambiguous? Specifically, when no key strongly matches the query, does the attention head minimize the maximum attention mass placed on any single distractor (i.e., spread attention uniformly) rather than collapsing onto a spuriously similar distractor?

## Setup

**Synthetic generator only.** No trained models required.

- **Vocabulary**: 4 token types — `TARGET`, `DISTRACTOR_A`, `DISTRACTOR_B`, `DISTRACTOR_C`.
- **Sequence structure**: Each sequence has one query position (last token) and 3 key positions (first three tokens). The key positions contain exactly one of each distractor type; the target is **never present** in keys. This forces the "no good match" regime.
- **Embeddings**: Fixed random embeddings for each token type (deterministic per seed). Embedding dimension `d_model = 32`.
- **Query construction**: The query embedding is a convex combination `q = α·e_TARGET + (1-α)·e_NOISE`, where `e_NOISE` is a fixed random vector orthogonal to the **TARGET** embedding (so `dot(e_NOISE, e_TARGET) = 0`). The sweep parameter `α ∈ [0, 1]` therefore cleanly controls target similarity: at `α = 0` the query has exactly zero target similarity. `e_NOISE` deliberately retains small **incidental** similarity to the distractors — that spurious match is precisely what a minimax head must avoid collapsing onto. (It is *not* orthogonal to the distractors; a query orthogonal to every key would make the spreading question trivial.)
- **Attention computation**: Standard scaled dot-product attention. The model function receives `(query, keys)` and returns attention weights over the 3 key positions (softmax output).

**Canonical measurement condition**: `α = 0.0` (pure noise query, zero target similarity; only incidental, spurious distractor similarity remains). At this condition, the minimax-optimal attention is uniform `[1/3, 1/3, 1/3]`, achieving the minimum possible maximum weight on any distractor (= 1/3) — this holds unconditionally, since uniform minimizes the max of any probability vector. Any non-uniform distribution (e.g. one that collapses onto the spuriously-similar distractor) has max weight > 1/3.

## Canonical model function signature

```python
def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Args:
        query: shape (d_model,), the query vector for a single position
        keys:  shape (3, d_model), the three key vectors (distractors only)
    Returns:
        attn_weights: shape (3,), non-negative, sums to 1.0
    """
```

## Payload contract

`task.evaluate(model_fn)` returns a dict with exact keys:

```python
{
    "version": 1,                           # payload schema version
    "d_model": 32,                          # embedding dimension
    "sweep": [
        {
            "alpha": 0.0,                   # sweep parameter (float)
            "query": [...],                 # list[float], length d_model
            "keys": [[...], [...], [...]],  # list[3][d_model]
            "attn_weights": [...],          # list[float], length 3, sums to 1
            "max_weight": 0.45,             # float, max(attn_weights)
            "entropy": 1.05,                # float, -sum(p log p) in nats
            "uniform_kl": 0.02,             # float, KL(attn || uniform) in nats
        },
        ...                                 # one record per alpha in sweep
    ],
    "sweep_alphas": [0.0, 0.1, 0.2, ..., 1.0],  # canonical sweep, 11 points
}
```

- `query` and `keys` are included for reproducibility/debugging; `benchmark.score` does not read them.
- `max_weight` is the primary per-slice measurement: maximum attention mass on any single distractor.
- `entropy` and `uniform_kl` are supplementary diagnostics.

## Metrics

Returned by `benchmark.score(payload)` — flat dict of scalars:

| Metric | Formula | Direction | Interpretation |
|--------|---------|-----------|----------------|
| `version` | `1` | — | Benchmark version |
| `minimax_regret_canonical` | `max_weight(α=0) - 1/3` | **smaller is better** | Headline summary. Excess max-weight over minimax optimum at canonical condition. 0 = perfect minimax. |
| `minimax_regret_alpha_<val>` | `max_weight(α) - 1/3` | smaller is better | Per-slice regret. Key uses `0p0` format (e.g., `minimax_regret_alpha_0p0`). |
| `entropy_alpha_<val>` | `entropy` from payload | larger is better | Attention entropy at each α. Uniform = log(3) ≈ 1.099. |
| `uniform_kl_alpha_<val>` | `uniform_kl` from payload | smaller is better | KL divergence from uniform. 0 = uniform. |
| `linear_baseline_regret_canonical` | `max_weight_linear(α=0) - 1/3` | — | Reference: regret of a linear attention baseline (no softmax), computed identically. |
| `lift_over_linear_canonical` | `linear_baseline_regret_canonical - minimax_regret_canonical` | **larger is better** | Improvement over linear baseline. Positive = mechanism helps. |

**Baseline (linear attention)**: `attn_weights ∝ query @ keys.T` (no softmax, renormalized to sum=1). This is the "no-mechanism" reference — a linear map cannot implement the minimax spreading behavior.

## Bump procedure

- `VERSION` in `benchmark.py` increments on any change to metric formulas, payload keys, or canonical sweep values.
- `README.md` metrics table and payload contract updated in same commit.
- Old `benchmark.json` files retained; dashboard filters to highest version.