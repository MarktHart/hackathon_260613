import numpy as np
from dataclasses import dataclass
from typing import Callable
import math


@dataclass(frozen=True)
class Batch:
    """Container for the generated sequence pairs."""
    base_tokens: np.ndarray      # [n_pairs_total, seq_len] int32
    edited_tokens: np.ndarray    # [n_pairs_total, seq_len] int32 (padded to seq_len)
    edit_distances: np.ndarray   # [n_pairs_total] int32, true Levenshtein distance
    seq_len: int
    vocab_size: int


def _levenshtein(a: list[int], b: list[int]) -> int:
    """Compute Levenshtein edit distance between two token lists."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )
    return dp[m][n]


def _apply_edits(tokens: list[int], k: int, vocab_size: int, rng: np.random.Generator) -> list[int]:
    """Apply exactly k random edit operations to a token sequence."""
    result = tokens.copy()
    for _ in range(k):
        op = rng.choice(3)  # 0=insert, 1=delete, 2=substitute
        if op == 0:  # insert
            pos = rng.integers(0, len(result) + 1)
            new_tok = rng.integers(0, vocab_size)
            result.insert(pos, int(new_tok))
        elif op == 1:  # delete
            if len(result) > 1:
                pos = rng.integers(0, len(result))
                result.pop(pos)
        else:  # substitute
            if len(result) > 0:
                pos = rng.integers(0, len(result))
                new_tok = rng.integers(0, vocab_size)
                result[pos] = int(new_tok)
    return result


def _pad_or_truncate(seq: list[int], target_len: int, pad_token: int = 0) -> list[int]:
    """Pad or truncate sequence to target length."""
    if len(seq) >= target_len:
        return seq[:target_len]
    return seq + [pad_token] * (target_len - len(seq))


def generate(seed: int = 0) -> Batch:
    """
    Generate sequence pairs with known edit distances.
    
    Deterministic for a given seed. Canonical parameters are hardcoded
    to match the canonical measurement condition in README.md.
    """
    # Canonical parameters (must match README.md)
    seq_len = 32
    vocab_size = 100
    edit_distances = list(range(9))  # 0 through 8
    pairs_per_distance = 50
    
    rng = np.random.default_rng(seed)
    
    all_base = []
    all_edited = []
    all_dists = []
    
    for k in edit_distances:
        for _ in range(pairs_per_distance):
            # Generate base sequence
            base = rng.integers(0, vocab_size, size=seq_len).tolist()
            
            # Apply k edits
            edited = _apply_edits(base, k, vocab_size, rng)
            
            # Verify edit distance (should equal k, but compute to be sure)
            true_dist = _levenshtein(base, edited)
            
            # Pad/truncate both to seq_len
            base_padded = _pad_or_truncate(base, seq_len)
            edited_padded = _pad_or_truncate(edited, seq_len)
            
            all_base.append(base_padded)
            all_edited.append(edited_padded)
            all_dists.append(true_dist)
    
    return Batch(
        base_tokens=np.array(all_base, dtype=np.int32),
        edited_tokens=np.array(all_edited, dtype=np.int32),
        edit_distances=np.array(all_dists, dtype=np.int32),
        seq_len=seq_len,
        vocab_size=vocab_size
    )


def _attention_distance(attn1: np.ndarray, attn2: np.ndarray) -> float:
    """
    Compute 1 - cosine similarity between two attention matrices.
    
    attn1, attn2: [seq_len, seq_len] float32
    Returns scalar in [0, 2] (0 = identical, 2 = opposite)
    """
    flat1 = attn1.flatten()
    flat2 = attn2.flatten()
    
    norm1 = np.linalg.norm(flat1)
    norm2 = np.linalg.norm(flat2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # orthogonal by convention
    
    cos_sim = np.dot(flat1, flat2) / (norm1 * norm2)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim


def evaluate(model_fn: Callable[[np.ndarray], np.ndarray]) -> dict:
    """
    Run model_fn on base and edited sequences, compute attention distances.
    
    Args:
        model_fn: Callable taking [batch, seq_len] int32 tokens,
                  returning [batch, seq_len, seq_len] float32 attention weights
                  for the canonical layer/head.
    
    Returns:
        Payload dict matching the contract in README.md.
    """
    batch = generate(seed=42)  # fixed seed for canonical evaluation
    
    n_total = batch.base_tokens.shape[0]
    seq_len = batch.seq_len
    
    # Process in chunks to avoid OOM (model_fn handles its own batching)
    chunk_size = 64
    base_attns = []
    edited_attns = []
    
    for i in range(0, n_total, chunk_size):
        base_chunk = batch.base_tokens[i:i+chunk_size]
        edited_chunk = batch.edited_tokens[i:i+chunk_size]
        
        base_attn = model_fn(base_chunk)      # [chunk, seq_len, seq_len]
        edited_attn = model_fn(edited_chunk)  # [chunk, seq_len, seq_len]
        
        base_attns.append(base_attn)
        edited_attns.append(edited_attn)
    
    base_attns = np.concatenate(base_attns, axis=0)
    edited_attns = np.concatenate(edited_attns, axis=0)
    
    # Compute attention distances per pair
    attn_distances = np.array([
        _attention_distance(base_attns[i], edited_attns[i])
        for i in range(n_total)
    ], dtype=np.float32)
    
    # Aggregate by edit distance
    unique_dists = np.unique(batch.edit_distances)
    sweep = []
    for d in unique_dists:
        mask = batch.edit_distances == d
        dists_at_d = attn_distances[mask]
        sweep.append({
            "edit_distance": int(d),
            "attn_distance_mean": float(np.mean(dists_at_d)),
            "attn_distance_std": float(np.std(dists_at_d)),
            "n_pairs": int(np.sum(mask))
        })
    
    # Compute linear baseline: random attention patterns
    # Use a fixed seed so baseline is reproducible across attempts
    baseline_rng = np.random.default_rng(12345)
    n_pairs_per_dist = [s["n_pairs"] for s in sweep]
    baseline_means = []
    baseline_stds = []
    
    for n in n_pairs_per_dist:
        # Generate random attention matrices (uniform on simplex per row)
        # Shape: [n, seq_len, seq_len]
        rand_attn1 = baseline_rng.random((n, seq_len, seq_len), dtype=np.float32)
        rand_attn2 = baseline_rng.random((n, seq_len, seq_len), dtype=np.float32)
        
        # Normalize rows to sum to 1 (attention-like)
        rand_attn1 = rand_attn1 / (rand_attn1.sum(axis=-1, keepdims=True) + 1e-8)
        rand_attn2 = rand_attn2 / (rand_attn2.sum(axis=-1, keepdims=True) + 1e-8)
        
        baseline_dists = np.array([
            _attention_distance(rand_attn1[i], rand_attn2[i])
            for i in range(n)
        ])
        baseline_means.append(float(np.mean(baseline_dists)))
        baseline_stds.append(float(np.std(baseline_dists)))
    
    return {
        "version": 1,
        "model_name": "gpt2",
        "layer": 5,
        "head": 3,
        "seq_len": seq_len,
        "vocab_size": batch.vocab_size,
        "attention_distance_metric": "1_minus_cosine",
        "sweep": sweep,
        "linear_baseline": {
            "attn_distance_mean": baseline_means,
            "attn_distance_std": baseline_stds
        }
    }


def random_model_fn() -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a model_fn that outputs random attention weights of the correct shape.
    
    Pure NumPy, no torch, no GPU. Used for smoke testing.
    """
    def _fn(tokens: np.ndarray) -> np.ndarray:
        batch_size, seq_len = tokens.shape
        # Random attention: uniform on simplex per query position
        rng = np.random.default_rng(0)  # fixed seed for determinism in smoke test
        attn = rng.random((batch_size, seq_len, seq_len), dtype=np.float32)
        attn = attn / (attn.sum(axis=-1, keepdims=True) + 1e-8)
        return attn
    return _fn