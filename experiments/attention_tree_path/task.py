import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray                    # (batch_size, seq_len) int32
    target_pos: np.ndarray                # (batch_size, seq_len) int32, -1 = no target
    depth: int
    path_rule: str
    tree_type: str = "binary"
    node_ids: np.ndarray = None           # (batch_size, seq_len) int32, original node ids
    parent_ids: np.ndarray = None         # (batch_size, seq_len) int32
    depths: np.ndarray = None             # (batch_size, seq_len) int32
    is_leaf: np.ndarray = None            # (batch_size, seq_len) bool


# ──────────────────────────────────────────────────────────────────────
# Tree generation utilities
# ──────────────────────────────────────────────────────────────────────

def _build_binary_tree(depth: int) -> Tuple[List[int], List[int], List[int], List[bool]]:
    """
    Build a full binary tree of given depth (root depth=0).
    Returns (node_ids, parent_ids, depths, is_leaf) in pre-order traversal order.
    Node IDs are 0..N-1 in pre-order.
    """
    node_ids = []
    parent_ids = []
    depths = []
    is_leaf = []

    def dfs(node_id: int, parent_id: int, d: int):
        node_ids.append(node_id)
        parent_ids.append(parent_id)
        depths.append(d)
        if d == depth:
            is_leaf.append(True)
            return
        is_leaf.append(False)
        left_id = len(node_ids)  # next available id
        dfs(left_id, node_id, d + 1)
        right_id = len(node_ids)
        dfs(right_id, node_id, d + 1)

    dfs(0, -1, 0)
    return node_ids, parent_ids, depths, is_leaf


def _compute_target_positions(
    node_ids: List[int],
    parent_ids: List[int],
    depths: List[int],
    is_leaf: List[bool],
    path_rule: str,
) -> List[int]:
    """
    For each position in pre-order, compute the target position index for the given path_rule.
    Returns list of target indices (same length), -1 if no valid target.
    """
    n = len(node_ids)
    # Build maps: node_id -> position in pre-order
    node_to_pos = {nid: i for i, nid in enumerate(node_ids)}
    # Build children lists
    children = {nid: [] for nid in node_ids}
    for i, pid in enumerate(parent_ids):
        if pid != -1:
            children[pid].append(node_ids[i])

    targets = []
    for i in range(n):
        nid = node_ids[i]
        pid = parent_ids[i]
        d = depths[i]
        leaf = is_leaf[i]

        if path_rule == "ancestor_1":
            if pid == -1:
                targets.append(-1)
            else:
                targets.append(node_to_pos[pid])
        elif path_rule == "ancestor_2":
            if pid == -1 or parent_ids[node_to_pos[pid]] == -1:
                targets.append(-1)
            else:
                gp = parent_ids[node_to_pos[pid]]
                targets.append(node_to_pos[gp])
        elif path_rule == "descendant_leftmost":
            if leaf:
                targets.append(-1)
            else:
                # Find leftmost descendant leaf
                cur = nid
                while not is_leaf[node_to_pos[cur]]:
                    cur = children[cur][0]  # left child
                targets.append(node_to_pos[cur])
        elif path_rule == "sibling_next":
            if pid == -1:
                targets.append(-1)
            else:
                sibs = children[pid]
                try:
                    idx = sibs.index(nid)
                    if idx + 1 < len(sibs):
                        targets.append(node_to_pos[sibs[idx + 1]])
                    else:
                        targets.append(-1)
                except ValueError:
                    targets.append(-1)
        elif path_rule == "root":
            if nid == 0:
                targets.append(-1)
            else:
                targets.append(node_to_pos[0])
        else:
            raise ValueError(f"Unknown path_rule: {path_rule}")

    return targets


def generate(seed: int = 0) -> Batch:
    """
    Deterministic generation for a given seed.
    The canonical condition ignores seed (fixed tree), but we accept it for interface compliance.
    Returns a single Batch at the canonical condition (depth=3, ancestor_1).
    """
    # Canonical condition
    depth = 3
    path_rule = "ancestor_1"
    batch_size = 32

    node_ids, parent_ids, depths, is_leaf = _build_binary_tree(depth)
    seq_len = len(node_ids)
    targets = _compute_target_positions(node_ids, parent_ids, depths, is_leaf, path_rule)

    # Replicate for batch
    tokens = np.tile(np.array(node_ids, dtype=np.int32), (batch_size, 1))
    target_pos = np.tile(np.array(targets, dtype=np.int32), (batch_size, 1))
    node_ids_arr = np.tile(np.array(node_ids, dtype=np.int32), (batch_size, 1))
    parent_ids_arr = np.tile(np.array(parent_ids, dtype=np.int32), (batch_size, 1))
    depths_arr = np.tile(np.array(depths, dtype=np.int32), (batch_size, 1))
    is_leaf_arr = np.tile(np.array(is_leaf, dtype=bool), (batch_size, 1))

    return Batch(
        tokens=tokens,
        target_pos=target_pos,
        depth=depth,
        path_rule=path_rule,
        tree_type="binary",
        node_ids=node_ids_arr,
        parent_ids=parent_ids_arr,
        depths=depths_arr,
        is_leaf=is_leaf_arr,
    )


def _generate_sweep_batch(depth: int, path_rule: str, batch_size: int = 32) -> Batch:
    """Generate a batch for a specific sweep condition."""
    node_ids, parent_ids, depths, is_leaf = _build_binary_tree(depth)
    seq_len = len(node_ids)
    targets = _compute_target_positions(node_ids, parent_ids, depths, is_leaf, path_rule)

    tokens = np.tile(np.array(node_ids, dtype=np.int32), (batch_size, 1))
    target_pos = np.tile(np.array(targets, dtype=np.int32), (batch_size, 1))
    node_ids_arr = np.tile(np.array(node_ids, dtype=np.int32), (batch_size, 1))
    parent_ids_arr = np.tile(np.array(parent_ids, dtype=np.int32), (batch_size, 1))
    depths_arr = np.tile(np.array(depths, dtype=np.int32), (batch_size, 1))
    is_leaf_arr = np.tile(np.array(is_leaf, dtype=bool), (batch_size, 1))

    return Batch(
        tokens=tokens,
        target_pos=target_pos,
        depth=depth,
        path_rule=path_rule,
        tree_type="binary",
        node_ids=node_ids_arr,
        parent_ids=parent_ids_arr,
        depths=depths_arr,
        is_leaf=is_leaf_arr,
    )


def _compute_correct_attention(
    attn_weights: np.ndarray,  # (batch, n_heads, seq_len, seq_len)
    target_pos: np.ndarray,    # (batch, seq_len)
) -> Tuple[float, float, int]:
    """
    Compute mean and std of attention weight on correct target positions.
    Only averages over valid queries (target_pos != -1).
    Returns (mean, std, n_valid_queries).
    """
    batch_size, n_heads, seq_len, _ = attn_weights.shape
    # Average over heads first: (batch, seq_len, seq_len)
    attn_mean_heads = attn_weights.mean(axis=1)

    valid_mask = target_pos != -1
    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        return 0.0, 0.0, 0

    # Gather attention weights at target positions
    # attn_mean_heads[b, q, target_pos[b, q]] for each valid (b, q)
    batch_idx, query_idx = np.where(valid_mask)
    target_idx = target_pos[valid_mask]
    correct_attn = attn_mean_heads[batch_idx, query_idx, target_idx]

    return float(correct_attn.mean()), float(correct_attn.std()), n_valid


def _compute_per_head_attention(
    attn_weights: np.ndarray,  # (batch, n_heads, seq_len, seq_len)
    target_pos: np.ndarray,    # (batch, seq_len)
) -> List[Dict[str, float]]:
    """Compute per-head mean/std at canonical condition."""
    batch_size, n_heads, seq_len, _ = attn_weights.shape
    valid_mask = target_pos != -1
    batch_idx, query_idx = np.where(valid_mask)
    target_idx = target_pos[valid_mask]

    results = []
    for h in range(n_heads):
        correct_attn = attn_weights[batch_idx, h, query_idx, target_idx]
        results.append({
            "head": h,
            "correct_attn_mean": float(correct_attn.mean()),
            "correct_attn_std": float(correct_attn.std()),
        })
    return results


# ──────────────────────────────────────────────────────────────────────
# Model function signature
# ──────────────────────────────────────────────────────────────────────

ModelFn = callable  # Batch -> np.ndarray[float, (batch, n_heads, seq_len, seq_len)]


def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """
    Run model_fn over the canonical batch and all sweep conditions.
    Returns payload dict matching benchmark.score expectations.
    """
    # Canonical batch
    canon_batch = generate(seed=0)
    canon_attn = model_fn(canon_batch)
    # Validate shape
    batch_size, n_heads, seq_len, _ = canon_attn.shape
    assert n_heads == 4, f"Expected 4 heads, got {n_heads}"
    assert seq_len == 15, f"Expected seq_len 15, got {seq_len}"

    canon_mean, canon_std, canon_n = _compute_correct_attention(canon_attn, canon_batch.target_pos)
    head_slice = _compute_per_head_attention(canon_attn, canon_batch.target_pos)

    # Sweep conditions
    sweep_configs = [
        {"depth": 2, "path_rule": "ancestor_1"},
        {"depth": 3, "path_rule": "ancestor_1"},
        {"depth": 4, "path_rule": "ancestor_1"},
        {"depth": 3, "path_rule": "ancestor_2"},
        {"depth": 3, "path_rule": "descendant_leftmost"},
        {"depth": 3, "path_rule": "sibling_next"},
    ]

    sweep = []
    for cfg in sweep_configs:
        batch = _generate_sweep_batch(cfg["depth"], cfg["path_rule"])
        attn = model_fn(batch)
        mean, std, n_valid = _compute_correct_attention(attn, batch.target_pos)
        sweep.append({
            "depth": cfg["depth"],
            "path_rule": cfg["path_rule"],
            "correct_attn_mean": mean,
            "correct_attn_std": std,
            "n_valid_queries": n_valid,
        })

    return {
        "version": 1,
        "config": {
            "tree_type": "binary",
            "depth": 3,
            "path_rule": "ancestor_1",
            "batch_size": 32,
            "seq_len": 15,
            "n_heads": 4,
        },
        "sweep": sweep,
        "head_slice": head_slice,
    }


# ──────────────────────────────────────────────────────────────────────
# Random model function for smoke test
# ──────────────────────────────────────────────────────────────────────

def random_model_fn() -> ModelFn:
    """
    Returns a model_fn that outputs uniform random attention weights
    (normalized to sum to 1 over key positions).
    Pure NumPy, no torch, no GPU.
    """
    def _fn(batch: Batch) -> np.ndarray:
        batch_size = batch.tokens.shape[0]
        seq_len = batch.tokens.shape[1]
        n_heads = 4
        # Uniform random then softmax (but uniform Dirichlet is simpler)
        # Use Dirichlet(1) = uniform over simplex
        noise = np.random.exponential(scale=1.0, size=(batch_size, n_heads, seq_len, seq_len))
        attn = noise / noise.sum(axis=-1, keepdims=True)
        return attn.astype(np.float32)
    return _fn