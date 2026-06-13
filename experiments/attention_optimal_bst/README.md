# Goal: attention_optimal_bst

## Question

Can a transformer's attention mechanism implement **optimal binary search tree (BST) search** — attending to the exact sequence of nodes that an optimal BST search would visit, given keys stored in the context and a query key?

## Setup

**Synthetic generator.** No trained model. We generate sequences that encode:
- A set of `n` keys with associated values, arranged as an **optimal BST** (minimizing expected search cost given access probabilities)
- A **query key** to search for
- Positional structure that lets attention "navigate" the tree

The generator produces a batch of independent search episodes. Each episode is a token sequence:
```
[ROOT] [LEFT_CHILD] [RIGHT_CHILD] ... [QUERY] [SEARCH_TRACE_POSITIONS]
```
where the search trace positions are reserved slots where the model *should* attend if it correctly follows the optimal BST search path from root to the query key (or to the leaf where it would be inserted).

**Model function contract** (what attempts must provide):
```python
def model_fn(tokens: np.ndarray,          # shape (B, T), int32 token IDs
             ) -> np.ndarray:             # shape (B, H, T, T), float32 attention weights
    """
    Return attention weights for all heads. No gradients, no training.
    Batch dimension B, head dimension H, query position dimension T, key position dimension T.
    Rows should be a valid attention distribution (each (b, h, q, :) sums to 1).
    """
```

`task.py` also exports `random_model_fn() -> model_fn`: a zero-dependency
**factory** (pure NumPy) that returns a `model_fn` with exactly the signature
above whose body emits the uniform-over-key-nodes baseline. The pipeline uses
it for the smoke test
(`benchmark.score(task.evaluate(task.random_model_fn()))`) and `benchmark.py`
treats this uniform distribution as the no-mechanism `linear_baseline`.

## Canonical Measurement Condition

- **Tree size**: `n = 15` keys. The Zipf(1.2)-skewed optimal BST is unbalanced, so search paths run **1–6 nodes** long (tree depth up to 5 edges); the per-slice `pathlen_<L>` metrics cover `L = 1..6`.
- **Sequence length**: `T = 32` (15 key nodes + 1 query + 16 trace positions)
- **Heads**: `H = 4` (we measure the best head)
- **Key distribution**: Keys `0..14`, access probabilities `zipf(1.2)` → optimal BST via Knuth's DP
- **Query distribution**: Uniform over `0..14` (present keys) + 5 absent keys `[-1, -2, -3, -4, -5]`
- **Batch size**: `B = 128` episodes
- **Seed**: `0` (fixed; `generate` ignores seed but accepts it for API compatibility)

The canonical condition is **fully fixed**. `generate(seed)` returns the same batch for any seed.

## Payload Contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,
    "canonical_condition": {
        "n_keys": 15,
        "seq_len": 32,
        "n_heads": 4,
        "batch_size": 128,
        "key_distribution": "zipf_1p2",
        "query_distribution": "uniform_present_plus_5_absent"
    },
    "sweep": [
        {
            "query_key": int,           # the key being searched for (-5..14)
            "optimal_path": list[int],  # token indices of optimal BST search path (root→target)
            "attn_to_path": list[float],# per-head max attention mass on each path position
            "path_length": int,         # len(optimal_path)
            "head_idx": int,            # best head index (0..H-1)
        },
        ...  # one record per episode in the batch (128 records)
    ],
    "aggregated": {
        "best_head": int,                 # head with highest mean path attention
        "mean_path_attention": float,     # mean over episodes of mean(attn_to_path) for best head
        "perfect_episodes": int,          # episodes where best head put >0.5 mass on EVERY path position
        "total_episodes": int,            # 128
        "mean_path_completion_rate": float,  # mean over episodes of (path positions with attn>0.5)/path_length
    }
}
```

**Semantics:**
- `optimal_path`: Token indices (0..T-1) of nodes visited by optimal BST search for this query.
- `attn_to_path`: For the best head, the maximum attention weight on each position in `optimal_path` when querying from the **final trace position** (the "answer" position). Length matches `path_length`.
- `perfect_episodes`: Count of episodes where the best head attends >0.5 to *every* node on the optimal path.

## Metrics

`benchmark.score(payload)` returns a flat dict:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `bst_search_accuracy_canonical` | `perfect_episodes / total_episodes` | bigger better | Fraction of searches where attention perfectly traces optimal path |
| `bst_mean_path_attention_canonical` | `mean_path_attention` | bigger better | Average attention mass on optimal path nodes |
| `bst_path_completion_rate_canonical` | Mean over episodes of `(positions_with_attn>0.5) / path_length` | bigger better | Fraction of path steps correctly attended |
| `bst_best_head_canonical` | `best_head` | — | Which head (0..3) performed best |
| `bst_search_accuracy_pathlen_<L>` | Fraction of episodes with optimal path length `L` that are "perfect" | bigger better | Per-slice accuracy by search depth (L = 1..6) |
| `linear_baseline_bst_search_accuracy_canonical` | Baseline: uniform attention over all key nodes → `0.0` | bigger better | Random-guessing baseline |
| `linear_baseline_mean_path_attention_canonical` | `1 / n_keys` | bigger better | Baseline mass per node |
| `lift_over_linear_baseline_bst_search_accuracy` | `bst_search_accuracy_canonical - linear_baseline_bst_search_accuracy_canonical` | bigger better | Accuracy improvement over baseline |
| `lift_over_linear_baseline_mean_path_attention` | `bst_mean_path_attention_canonical - 1/n_keys` | bigger better | Path-mass improvement over baseline |

**Headline metric:** `bst_search_accuracy_canonical` — the fraction of search
episodes where the best head's attention perfectly traces the optimal BST path.

**Per-slice metrics:** `bst_search_accuracy_pathlen_<L>` exposes where a method
holds vs. breaks as a function of search depth (deeper paths are harder).

**Baseline computation:** For each episode, uniform attention over the `n_keys = 15`
key nodes gives `1/15 ≈ 0.067` mass per node. Since no node clears the `>0.5`
threshold, baseline `bst_search_accuracy = 0`, `mean_path_attention = 1/15`,
`path_completion_rate = 0`. `random_model_fn()` realises exactly this baseline.

## Bump Procedure

- `VERSION` in `benchmark.py` increments when:
  - Canonical condition changes (n_keys, seq_len, distributions)
  - Payload keys added/removed/renamed
  - Metric formulas change
- Adding a new metric without changing existing ones does **not** require a version bump.
- After bumping, update this README's "Payload Contract" and "Metrics" sections in the same commit.

## Optional Pipeline Hooks

- `GPU_REQUIREMENT = 1` (default; attempts run on GPU)
- `is_obviously_broken(metrics)` returns `True` (skip the jury) if:
  - Any metric is NaN or inf
  - `bst_mean_path_attention_canonical <= linear_baseline_mean_path_attention_canonical`
    (the attempt does not beat uniform attention — no tree navigation at all)

  It never returns `True` for a borderline-but-real result; the uniform
  baseline itself sits exactly at the threshold (smoke test only).