# Attention Tree Path

## Question

Can attention heads in a transformer reliably trace compositional paths through a tree-structured dependency graph? Specifically, given a sequence where tokens are arranged in a tree hierarchy (each node has a parent, leaves have no children), do attention heads learn to attend from a node to its ancestors, descendants, or siblings along the tree structure — and does this behavior generalize across tree depths and branching factors?

## Setup

**Synthetic generator only.** No trained model. The task generates sequences with explicit tree-structured dependencies:

- Each sequence is a linearization of a rooted tree (pre-order traversal).
- Each token carries a `node_id`, `parent_id`, `depth`, and `is_leaf` flag.
- The "ground truth" attention target for each query position is defined by a **path rule**:
  - `ancestor_k`: attend to the k-th ancestor (k=1 → parent, k=2 → grandparent, etc.)
  - `descendant_leftmost`: attend to the leftmost descendant leaf
  - `sibling_next`: attend to the next sibling in pre-order
  - `root`: attend to the root node

The generator produces batches of such sequences with fixed vocabulary (node IDs mapped to tokens). The canonical measurement condition uses:
- Tree: full binary tree, depth = 3 (15 nodes)
- Sequence length: 15 (pre-order traversal)
- Path rule: `ancestor_1` (parent)
- Number of sequences per batch: 32
- Evaluation: mean attention weight placed on the correct target position, averaged over all query positions that have a valid target (root has no parent, so excluded)

**Canonical measurement condition** (every attempt must use):
- `tree_type`: "binary"
- `depth`: 3
- `path_rule`: "ancestor_1"
- `batch_size`: 32
- `seed`: 0 (for generate)
- `model_fn` receives a `Batch` and returns `attn_weights: np.ndarray[float, (batch, n_heads, seq_len, seq_len)]`

## Payload Contract

`task.evaluate(model_fn)` returns a dict with exactly these keys:

```python
{
    "version": 1,                          # matches benchmark.VERSION
    "config": {
        "tree_type": "binary",
        "depth": 3,
        "path_rule": "ancestor_1",
        "batch_size": 32,
        "seq_len": 15,
        "n_heads": 4,                      # fixed by canonical model_fn signature
    },
    "sweep": [
        {
            "depth": 2,
            "path_rule": "ancestor_1",
            "correct_attn_mean": 0.0,      # filled by evaluate
            "correct_attn_std": 0.0,
            "n_valid_queries": 0,
        },
        {
            "depth": 3,
            "path_rule": "ancestor_1",
            "correct_attn_mean": 0.0,
            "correct_attn_std": 0.0,
            "n_valid_queries": 0,
        },
        {
            "depth": 4,
            "path_rule": "ancestor_1",
            "correct_attn_mean": 0.0,
            "correct_attn_std": 0.0,
            "n_valid_queries": 0,
        },
        {
            "depth": 3,
            "path_rule": "ancestor_2",
            "correct_attn_mean": 0.0,
            "correct_attn_std": 0.0,
            "n_valid_queries": 0,
        },
        {
            "depth": 3,
            "path_rule": "descendant_leftmost",
            "correct_attn_mean": 0.0,
            "correct_attn_std": 0.0,
            "n_valid_queries": 0,
        },
        {
            "depth": 3,
            "path_rule": "sibling_next",
            "correct_attn_mean": 0.0,
            "correct_attn_std": 0.0,
            "n_valid_queries": 0,
        },
    ],
    "head_slice": [                        # per-head breakdown at canonical condition
        {"head": 0, "correct_attn_mean": 0.0, "correct_attn_std": 0.0},
        {"head": 1, "correct_attn_mean": 0.0, "correct_attn_std": 0.0},
        {"head": 2, "correct_attn_mean": 0.0, "correct_attn_std": 0.0},
        {"head": 3, "correct_attn_mean": 0.0, "correct_attn_std": 0.0},
    ],
}
```

All `correct_attn_*` values are **mean attention weight** (not logits, not entropy) on the ground-truth target position, averaged over valid query positions in the batch. Range `[0, 1]`. Higher = better path tracing.

## Metrics

`benchmark.score(payload)` returns a flat dict:

| Metric | Formula | Direction | Meaning |
|--------|---------|-----------|---------|
| `version` | — | — | Payload version |
| `tree_path_canonical` | `correct_attn_mean` at depth=3, ancestor_1 | Bigger better | Headline: how well the model traces parent links at canonical depth |
| `tree_path_depth_2` | `correct_attn_mean` at depth=2, ancestor_1 | Bigger better | Generalization to shallower trees |
| `tree_path_depth_3` | `correct_attn_mean` at depth=3, ancestor_1 | Bigger better | Canonical slice (duplicate of canonical for dropdown) |
| `tree_path_depth_4` | `correct_attn_mean` at depth=4, ancestor_1 | Bigger better | Generalization to deeper trees |
| `tree_path_ancestor_2` | `correct_attn_mean` at depth=3, ancestor_2 | Bigger better | Two-hop composition (grandparent) |
| `tree_path_descendant` | `correct_attn_mean` at depth=3, descendant_leftmost | Bigger better | Downward path tracing |
| `tree_path_sibling` | `correct_attn_mean` at depth=3, sibling_next | Bigger better | Lateral path tracing |
| `tree_path_robustness` | `min(tree_path_depth_2, tree_path_depth_3, tree_path_depth_4) / max(...)` | Bigger better | Depth generalization ratio ∈ [0,1] |
| `tree_path_head_best` | `max(head_slice.correct_attn_mean)` | Bigger better | Best single head at canonical condition |
| `tree_path_head_worst` | `min(head_slice.correct_attn_mean)` | Bigger better | Worst single head (consistency) |
| `tree_path_head_gap` | `tree_path_head_best - tree_path_head_worst` | Smaller better | Head specialization spread |
| `linear_baseline_canonical` | `1 / (seq_len - 1)` ≈ 0.071 | — | Uniform attention baseline (excludes self) |
| `lift_over_baseline` | `tree_path_canonical - linear_baseline_canonical` | Bigger better | Improvement over random attention |

**Baseline**: Uniform attention over all other positions → `1 / (seq_len - 1)`. At seq_len=15, this is 1/14 ≈ 0.0714.

## Bump Procedure

- `VERSION` in `benchmark.py` must be incremented when:
  - Any metric formula changes
  - Payload keys are added/removed/retyped
  - Canonical condition (depth, path_rule, tree_type) changes
  - Sweep structure changes incompatibly
- After bump: update this README's payload contract and metrics table in the same commit.
- Old `benchmark.json` files remain on disk; dashboard filters to highest version.

## Model Function Signature

```python
def model_fn(batch: Batch) -> np.ndarray:
    """
    Args:
        batch: Batch with fields
            - tokens: np.ndarray[int, (batch_size, seq_len)]
            - target_pos: np.ndarray[int, (batch_size, seq_len)]  # -1 for no valid target
            - depth: int
            - path_rule: str
    Returns:
        attn_weights: np.ndarray[float, (batch_size, n_heads, seq_len, seq_len)]
            Attention weights (post-softmax), summing to 1 over last dim.
    """
```

`n_heads` is fixed to 4 by the canonical condition. Attempts may ignore `batch.depth` and `batch.path_rule` (they are fixed per sweep record) but must handle variable `batch_size` and `seq_len`.