import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any, List, Optional

@dataclass(frozen=True)
class Batch:
    """Container for the generated evaluation sequences."""
    input_ids: np.ndarray          # [n_seq, seq_len]
    target_positions: np.ndarray   # [n_seq] position of target token
    source_positions: np.ndarray   # [n_seq] matching position in first pattern
    pattern_lengths: np.ndarray    # [n_seq]
    distances: np.ndarray          # [n_seq]
    target_tokens: np.ndarray      # [n_seq]

def generate(seed: int = 42) -> Batch:
    """
    Deterministically generate test sequences with repeated patterns.
    Seed controls pattern sampling and filler noise; canonical seed is 42.
    """
    rng = np.random.RandomState(seed)
    
    seq_len = 64
    vocab_size = 64
    BOS = 0
    PAD = 1
    pattern_lengths = [2, 3, 4]
    distances = [8, 16, 32]
    n_per_combo = 50
    
    n_seq = len(pattern_lengths) * len(distances) * n_per_combo
    input_ids = np.full((n_seq, seq_len), PAD, dtype=np.int32)
    target_positions = np.zeros(n_seq, dtype=np.int32)
    source_positions = np.zeros(n_seq, dtype=np.int32)
    pattern_lengths_arr = np.zeros(n_seq, dtype=np.int32)
    distances_arr = np.zeros(n_seq, dtype=np.int32)
    target_tokens = np.zeros(n_seq, dtype=np.int32)
    
    idx = 0
    for plen in pattern_lengths:
        for dist in distances:
            for _ in range(n_per_combo):
                # Build sequence: [BOS] [prefix] [pattern] [filler] [pattern] [target] [suffix]
                # Pattern starts at position 1 (after BOS)
                pattern_start_1 = 1
                pattern_start_2 = pattern_start_1 + plen + dist
                target_pos = pattern_start_2 + plen
                
                # Ensure we fit in seq_len
                if target_pos >= seq_len - 1:
                    # Fallback: place pattern earlier
                    pattern_start_1 = 1
                    pattern_start_2 = seq_len - plen - 2
                    target_pos = pattern_start_2 + plen
                    dist = pattern_start_2 - (pattern_start_1 + plen)
                
                # Sample pattern tokens from 2..vocab_size-1
                pattern = rng.randint(2, vocab_size, size=plen, dtype=np.int32)
                
                # Target token is a deterministic function of pattern (e.g., sum mod vocab)
                target_token = int((pattern.sum() + plen * 7) % (vocab_size - 2)) + 2
                
                # Fill prefix (before first pattern)
                prefix_len = pattern_start_1
                if prefix_len > 0:
                    input_ids[idx, :prefix_len] = rng.randint(2, vocab_size, size=prefix_len)
                input_ids[idx, 0] = BOS
                
                # First pattern occurrence
                input_ids[idx, pattern_start_1:pattern_start_1+plen] = pattern
                
                # Filler between patterns
                filler_len = dist
                if filler_len > 0:
                    filler_start = pattern_start_1 + plen
                    input_ids[idx, filler_start:filler_start+filler_len] = rng.randint(2, vocab_size, size=filler_len)
                
                # Second pattern occurrence
                input_ids[idx, pattern_start_2:pattern_start_2+plen] = pattern
                
                # Target token
                input_ids[idx, target_pos] = target_token
                
                # Suffix (remaining positions)
                suffix_start = target_pos + 1
                if suffix_start < seq_len:
                    input_ids[idx, suffix_start:] = rng.randint(2, vocab_size, size=seq_len - suffix_start)
                
                # Source position: corresponding token in first pattern (last token of pattern)
                source_pos = pattern_start_1 + plen - 1
                
                target_positions[idx] = target_pos
                source_positions[idx] = source_pos
                pattern_lengths_arr[idx] = plen
                distances_arr[idx] = dist
                target_tokens[idx] = target_token
                idx += 1
    
    return Batch(
        input_ids=input_ids[:idx],
        target_positions=target_positions[:idx],
        source_positions=source_positions[:idx],
        pattern_lengths=pattern_lengths_arr[:idx],
        distances=distances_arr[:idx],
        target_tokens=target_tokens[:idx]
    )

def random_model_fn() -> Callable[[np.ndarray], Dict[str, Any]]:
    """
    Returns a dummy model_fn that produces valid-shaped random outputs.
    Used for smoke testing the pipeline.
    """
    def _fn(input_ids: np.ndarray) -> Dict[str, Any]:
        batch, seq_len = input_ids.shape
        # Random attention: uniform over sequence for each layer/head
        n_layers = 12
        n_heads = 12
        attn = np.ones((n_layers, n_heads, seq_len, seq_len), dtype=np.float32) / seq_len
        # Random logits
        vocab_size = 64
        logits = np.random.randn(seq_len, vocab_size).astype(np.float32)
        return {
            "attn_weights": attn,
            "logits": logits
        }
    return _fn

def evaluate(model_fn: Callable[[np.ndarray], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Run model_fn on all generated sequences, extract per-head attention at target positions,
    and compile the payload for benchmark.score.
    """
    batch = generate(seed=42)  # Canonical seed
    n_seq = batch.input_ids.shape[0]
    
    sweep_records = []
    
    for i in range(n_seq):
        # Run model on single sequence (batch=1)
        input_ids = batch.input_ids[i:i+1]  # [1, seq_len]
        out = model_fn(input_ids)
        
        attn_weights = out["attn_weights"]  # [n_layers, n_heads, seq, seq]
        logits = out.get("logits", None)    # [seq, vocab] or None
        
        n_layers, n_heads, seq_len, _ = attn_weights.shape
        target_pos = int(batch.target_positions[i])
        source_pos = int(batch.source_positions[i])
        target_token = int(batch.target_tokens[i])
        
        # Attention from target_pos to all positions, for each layer/head
        # Shape: [n_layers, n_heads, seq_len]
        attn_from_target = attn_weights[:, :, target_pos, :]
        
        # Attention weight to source position
        attn_to_source_per_head = attn_from_target[:, :, source_pos]  # [n_layers, n_heads]
        
        # Max attention to any other position (excluding source and target itself)
        mask = np.ones(seq_len, dtype=bool)
        mask[source_pos] = False
        mask[target_pos] = False
        max_attn_elsewhere_per_head = attn_from_target[:, :, mask].max(axis=-1)  # [n_layers, n_heads]
        
        # Argmax attention position per head
        argmax_pos_per_head = attn_from_target.argmax(axis=-1)  # [n_layers, n_heads]
        correct_top1_per_head = (argmax_pos_per_head == source_pos)
        
        # Find best head (max attn_to_source)
        best_head_idx = np.unravel_index(attn_to_source_per_head.argmax(), (n_layers, n_heads))
        best_layer, best_head = best_head_idx
        
        attn_to_source = float(attn_to_source_per_head[best_layer, best_head])
        max_attn_elsewhere = float(max_attn_elsewhere_per_head[best_layer, best_head])
        correct_top1 = bool(correct_top1_per_head[best_layer, best_head])
        
        # Token prediction if logits provided
        predicted_token = None
        if logits is not None:
            predicted_token = int(logits[target_pos].argmax())
        
        sweep_records.append({
            "pattern_length": int(batch.pattern_lengths[i]),
            "distance": int(batch.distances[i]),
            "seq_idx": i % 50,
            "target_pos": target_pos,
            "source_pos": source_pos,
            "attn_to_source": attn_to_source,
            "max_attn_elsewhere": max_attn_elsewhere,
            "correct_top1": correct_top1,
            "target_token": target_token,
            "predicted_token": predicted_token if predicted_token is not None else -1
        })
    
    return {
        "version": 1,
        "config": {
            "seq_len": 64,
            "vocab_size": 64,
            "pattern_lengths": [2, 3, 4],
            "distances": [8, 16, 32],
            "n_per_combo": 50,
            "seed": 42
        },
        "sweep": sweep_records
    }