"""Synthetic BST search task for attention interpretability."""

from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import numpy as np


@dataclass(frozen=True)
class Batch:
    """One batch of BST search episodes."""
    tokens: np.ndarray              # (B, T) int32 token IDs
    optimal_paths: List[List[int]]  # per episode: token indices of optimal search path
    query_keys: List[int]           # per episode: the key being searched for
    path_lengths: List[int]         # per episode: len(optimal_paths[i])
    n_keys: int                     # 15
    seq_len: int                    # 32
    n_heads: int                    # 4


def _build_optimal_bst(keys: List[int], probs: List[float]) -> Tuple[Dict[int, Tuple[int, int]], int]:
    """
    Knuth's DP for optimal BST.
    Returns (tree, root_key) where tree maps
        key -> (left_child_key, right_child_key), -1 meaning no child,
    and root_key is the key at the root of the optimal tree.
    Keys are sorted ascending.
    """
    n = len(keys)
    # dp[i][j] = min cost for keys[i..j]
    dp = [[0.0] * (n + 1) for _ in range(n + 1)]
    root = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(n):
        dp[i][i + 1] = probs[i]
        root[i][i + 1] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length
            dp[i][j] = float('inf')
            sum_prob = sum(probs[i:j])
            for r in range(root[i][j - 1], min(root[i + 1][j] + 1, j)):
                cost = dp[i][r] + dp[r + 1][j] + sum_prob
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    root[i][j] = r

    # Build tree structure
    tree = {}
    def build(i: int, j: int, parent: int = -1) -> int:
        if i >= j:
            return -1
        r = root[i][j]
        key = keys[r]
        left = build(i, r, key)
        right = build(r + 1, j, key)
        tree[key] = (left, right)
        return key

    root_key = build(0, n)
    return tree, root_key


def _search_path(tree: Dict[int, Tuple[int, int]], root_key: int, query: int) -> List[int]:
    """Return list of keys visited during BST search for query."""
    path = []
    current = root_key
    while current != -1:
        path.append(current)
        left, right = tree[current]
        if query < current:
            current = left
        elif query > current:
            current = right
        else:
            break  # found
    return path


def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch generation. Seed is accepted for API compatibility but ignored
    because the canonical condition is fully fixed.
    """
    # Canonical parameters
    n_keys = 15
    seq_len = 32
    n_heads = 4
    batch_size = 128

    # Keys 0..14 with Zipf(1.2) access probabilities
    keys = list(range(n_keys))
    alpha = 1.2
    raw_probs = np.array([1.0 / ((k + 1) ** alpha) for k in keys])
    probs = raw_probs / raw_probs.sum()

    # Build optimal BST (Knuth's DP). root_key is the actual root of the tree.
    tree, root_key = _build_optimal_bst(keys, probs.tolist())

    # Token layout:
    # 0..14: key nodes (token ID = key + 1, so 1..15)
    # 15: query token (token ID = 16 + query_key_offset)
    # 16..31: trace positions (token ID = 17..32)
    # We'll place query at position 15, trace positions 16..31

    # Queries: all 15 present keys + 5 absent keys (-1..-5)
    present_queries = keys
    absent_queries = [-1, -2, -3, -4, -5]
    all_queries = present_queries + absent_queries  # 20 queries
    # Repeat to get 128 episodes
    queries = (all_queries * (batch_size // len(all_queries) + 1))[:batch_size]

    # Build batch
    tokens = np.zeros((batch_size, seq_len), dtype=np.int32)
    optimal_paths = []
    path_lengths = []

    for b, q in enumerate(queries):
        # Key nodes at positions 0..14
        for k in keys:
            tokens[b, k] = k + 1  # token IDs 1..15

        # Query at position 15
        # Encode query: present keys 0..14 -> token 17..31, absent -1..-5 -> token 33..37
        if q >= 0:
            query_token = 17 + q
        else:
            query_token = 33 + (-q - 1)
        tokens[b, 15] = query_token

        # Trace positions 16..31 get unique tokens 50..65 (not attended to by design)
        for t in range(16, seq_len):
            tokens[b, t] = 50 + (t - 16)

        # Compute optimal search path (as token positions 0..14)
        key_path = _search_path(tree, root_key, q)
        # Convert keys to token positions (key k is at position k)
        path_positions = [k for k in key_path]
        optimal_paths.append(path_positions)
        path_lengths.append(len(path_positions))

    return Batch(
        tokens=tokens,
        optimal_paths=optimal_paths,
        query_keys=queries,
        path_lengths=path_lengths,
        n_keys=n_keys,
        seq_len=seq_len,
        n_heads=n_heads,
    )


def evaluate(model_fn) -> Dict[str, Any]:
    """
    Run model_fn over the canonical batch, compute attention-on-path metrics.
    Returns payload matching benchmark.score contract.
    """
    batch = generate()
    B, T = batch.tokens.shape
    H = batch.n_heads

    # Get attention weights: (B, H, T, T)
    attn = model_fn(batch.tokens)

    # Validate shape
    if attn.shape != (B, H, T, T):
        raise ValueError(f"model_fn returned shape {attn.shape}, expected ({B}, {H}, {T}, {T})")

    # For each episode, we look at attention FROM the last trace position (31)
    # TO the key positions (0..14). We measure a single global "best head":
    # the head with the highest mean (over episodes) path attention. Picking one
    # head across the whole batch — rather than a per-episode argmax — matches
    # the README ("we measure the best head") and avoids cherry-picking a
    # different head per episode, which would inflate the score.
    query_pos = T - 1  # position 31 (last trace position)

    # Attention from query_pos to key positions for every episode/head: (B, H, n_keys)
    attn_to_keys_all = attn[:, :, query_pos, :batch.n_keys]

    # head_means[h] = mean over episodes of mean(attn_to_path) for head h.
    head_means = np.zeros(H)
    for h in range(H):
        per_episode = np.empty(B)
        for b in range(B):
            path = batch.optimal_paths[b]
            if batch.path_lengths[b] > 0:
                per_episode[b] = attn_to_keys_all[b, h, path].mean()
            else:
                per_episode[b] = 0.0
        head_means[h] = per_episode.mean()

    best_head_overall = int(np.argmax(head_means))
    mean_path_attention = float(head_means[best_head_overall])

    # Build per-episode sweep records using the single global best head.
    sweep_records = []
    for b in range(B):
        path = batch.optimal_paths[b]
        path_len = batch.path_lengths[b]

        attn_on_path = (
            attn_to_keys_all[b, best_head_overall, path].tolist()
            if path_len > 0 else []
        )

        sweep_records.append({
            "query_key": batch.query_keys[b],
            "optimal_path": path,
            "attn_to_path": attn_on_path,
            "path_length": path_len,
            "head_idx": best_head_overall,
        })

    # Perfect episodes: best head puts >0.5 on EVERY path position
    perfect = 0
    path_completion_rates = []
    for b, rec in enumerate(sweep_records):
        attn_on_path = rec["attn_to_path"]
        path_len = rec["path_length"]
        if path_len == 0:
            path_completion_rates.append(1.0)
            continue
        n_above_half = sum(1 for a in attn_on_path if a > 0.5)
        path_completion_rates.append(n_above_half / path_len)
        if n_above_half == path_len:
            perfect += 1

    payload = {
        "version": 1,
        "canonical_condition": {
            "n_keys": batch.n_keys,
            "seq_len": batch.seq_len,
            "n_heads": batch.n_heads,
            "batch_size": B,
            "key_distribution": "zipf_1p2",
            "query_distribution": "uniform_present_plus_5_absent",
        },
        "sweep": sweep_records,
        "aggregated": {
            "best_head": best_head_overall,
            "mean_path_attention": mean_path_attention,
            "perfect_episodes": perfect,
            "total_episodes": B,
            "mean_path_completion_rate": float(np.mean(path_completion_rates)),
        },
    }
    return payload


def random_model_fn():
    """
    Factory: returns a callable with the same signature as a real model_fn.

    The returned model_fn takes tokens (B, T) and returns attention weights
    of shape (B, H, T, T). This baseline puts uniform attention over the key
    node positions (0..14) for every head and every query position — the
    no-mechanism reference. Pure NumPy, no torch, no GPU.
    """
    def model_fn(tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens)
        B, T = tokens.shape
        H = 4
        n_keys = 15

        # Uniform over key nodes (positions 0..14) for all query positions.
        key_mask = np.zeros(T, dtype=np.float32)
        key_mask[:n_keys] = 1.0 / n_keys

        attn = np.zeros((B, H, T, T), dtype=np.float32)
        attn[:, :, :, :] = key_mask[None, None, None, :]
        return attn

    return model_fn