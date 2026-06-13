# attention_bipartite

## Question
Does the model learn bipartite attention patterns — attending primarily *between* two distinct token groups rather than *within* each group — when the task structure demands it?

## Setup
**Synthetic generator.** We construct sequences with two semantically distinct token groups (Group A and Group B). The target requires the model to attend from each token in Group A to tokens in Group B (and vice versa), but never within the same group.

Each sequence has length `2 * group_size`. Positions `[0, group_size)` are Group A; positions `[group_size, 2*group_size)` are Group B.

The synthetic task: given a query token from one group, retrieve the matching key token from the *other* group (matching by a shared feature ID). This forces cross-group attention.

**Model function signature** (the contract between `task.py` and attempts):
```python
def model_fn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Args:
        q: (batch, seq_len, d_model) query projections
        k: (batch, seq_len, d_model) key projections
        v: (batch, seq_len, d_model) value projections
    Returns:
        attn_out: (batch, seq_len, d_model) attention output
    """
```

The attempt's `main.py` implementations this function using whatever mechanism it proposes (learned attention, sparse attention, etc.). `task.evaluate` calls this function on generated batches.

## Canonical measurement condition
- `group_size = 8` (sequence length 16)
- `d_model = 32`
- `num_features = 4` (each token has a feature ID in 0..3; matching is by feature ID)
- `batch_size = 32`
- Sweep axis: `num_heads ∈ [1, 2, 4, 8]` — tests whether multi-head structure helps isolate the bipartite pattern.

## Payload contract
`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": int,                    # benchmark.VERSION
    "config": {
        "group_size": int,
        "d_model": int,
        "num_features": int,
        "batch_size": int,
        "num_heads_sweep": list[int],
    },
    "sweep": [
        {
            "num_heads": int,
            "mean_attn_within": float,   # mean attention weight within same group (A→A + B→B) / 2
            "mean_attn_between": float,  # mean attention weight across groups (A→B + B→A) / 2
            "retrieval_acc": float,      # fraction of queries that attend max to correct cross-group key
        }
        for num_heads in config["num_heads_sweep"]
    ],
}
```

All attention weights are averaged over batch, sequence positions, and heads (for multi-head). `retrieval_acc` is 1.0 if every query's maximum attention weight lands on the correct cross-group key.

## Metrics
Computed by `benchmark.score(payload)`:

| Metric | Formula | Direction |
|--------|---------|-----------|
| `bipartite_score_canonical` | `mean_attn_between - mean_attn_within` at `num_heads=4` (canonical) | bigger better |
| `bipartite_score_num_heads_<h>` | same difference at each sweep point | bigger better |
| `retrieval_acc_num_heads_<h>` | retrieval accuracy at each sweep point | bigger better |
| `linear_baseline_bipartite_score_num_heads_<h>` | same metric for a linear (no softmax) baseline | — |
| `bipartite_robustness` | `min(bipartite_score_num_heads) / max(bipartite_score_num_heads)` across sweep | bigger better (in [0,1]) |
| `version` | `benchmark.VERSION` | — |

The headline metric is `bipartite_score_canonical`. A positive value means the model attends more between groups than within; negative means it fails the bipartite structure.

## Bump procedure
Bump `VERSION` in `benchmark.py` when:
- Any metric formula changes
- Payload keys are added/removed/renamed
- Canonical condition changes (e.g., group_size, d_model)
- Sweep axis values change

After bumping, update this README's "Payload contract" and "Metrics" tables in the same commit.