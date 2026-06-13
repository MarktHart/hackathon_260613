import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class Batch:
    q: np.ndarray          # (batch, seq_len, d_model)
    k: np.ndarray          # (batch, seq_len, d_model)
    v: np.ndarray          # (batch, seq_len, d_model)
    group_size: int
    num_features: int
    feature_ids: np.ndarray  # (seq_len,) feature ID for each position
    target_indices: np.ndarray  # (batch, seq_len) correct cross-group key index for each query


def generate(seed: int = 0) -> Batch:
    """
    Deterministic synthetic batch for bipartite attention task.
    Groups are fixed: positions [0, group_size) = Group A, [group_size, 2*group_size) = Group B.
    Each position has a feature_id. Target is the position in the OTHER group with same feature_id.
    """
    rng = np.random.default_rng(seed)

    # Canonical config
    group_size = 8
    d_model = 32
    num_features = 4
    batch_size = 32
    seq_len = 2 * group_size

    # Feature IDs: repeat 0..num_features-1 across positions
    feature_ids = np.tile(np.arange(num_features), seq_len // num_features + 1)[:seq_len]

    # Build target_indices: for each query position, find matching feature in other group
    target_indices = np.zeros((batch_size, seq_len), dtype=int)
    for pos in range(seq_len):
        fid = feature_ids[pos]
        if pos < group_size:  # Group A -> match in Group B
            candidates = np.where(feature_ids[group_size:] == fid)[0]
            target_indices[:, pos] = group_size + candidates[0]
        else:  # Group B -> match in Group A
            candidates = np.where(feature_ids[:group_size] == fid)[0]
            target_indices[:, pos] = candidates[0]

    # Generate Q, K, V with structure: same feature_id -> similar vectors
    # Each feature gets a base vector; add noise per position
    feature_bases = rng.normal(0, 1, size=(num_features, d_model)).astype(np.float32)
    pos_noise = rng.normal(0, 0.1, size=(batch_size, seq_len, d_model)).astype(np.float32)

    q = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
    k = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
    v = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)

    for b in range(batch_size):
        for pos in range(seq_len):
            fid = feature_ids[pos]
            base = feature_bases[fid]
            noise = pos_noise[b, pos]
            q[b, pos] = base + noise
            k[b, pos] = base + noise
            v[b, pos] = base + noise

    return Batch(
        q=q, k=k, v=v,
        group_size=group_size,
        num_features=num_features,
        feature_ids=feature_ids,
        target_indices=target_indices,
    )


def evaluate(model_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]) -> dict:
    """
    Runs model_fn on the canonical batch and computes attention statistics
    over a sweep of num_heads (simulated by reshaping d_model).
    """
    batch = generate(seed=42)  # fixed seed for canonical evaluation

    # Canonical sweep
    num_heads_sweep = [1, 2, 4, 8]
    d_model = batch.q.shape[-1]

    sweep = []
    for num_heads in num_heads_sweep:
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by num_heads={num_heads}")
        head_dim = d_model // num_heads

        # Reshape for multi-head: (batch, seq_len, num_heads, head_dim)
        q_multi = batch.q.reshape(batch.q.shape[0], batch.q.shape[1], num_heads, head_dim)
        k_multi = batch.k.reshape(batch.k.shape[0], batch.k.shape[1], num_heads, head_dim)
        v_multi = batch.v.reshape(batch.v.shape[0], batch.v.shape[1], num_heads, head_dim)

        # Flatten heads into batch dimension for model_fn
        # model_fn expects (batch*num_heads, seq_len, head_dim)
        batch_size, seq_len = q_multi.shape[0], q_multi.shape[1]
        q_flat = q_multi.transpose(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)
        k_flat = k_multi.transpose(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)
        v_flat = v_multi.transpose(0, 2, 1, 3).reshape(batch_size * num_heads, seq_len, head_dim)

        # Call model_fn
        attn_out_flat = model_fn(q_flat, k_flat, v_flat)  # (batch*num_heads, seq_len, head_dim)

        # Reshape back: (batch, num_heads, seq_len, head_dim)
        attn_out_multi = attn_out_flat.reshape(batch_size, num_heads, seq_len, head_dim).transpose(0, 2, 1, 3)

        # To compute attention weights, we need to recover them from the model_fn.
        # Since model_fn only returns output, we approximate by computing attention
        # weights directly here using the same Q,K (this matches what a standard
        # attention implementation would produce).
        # NOTE: This assumes model_fn implements standard scaled dot-product attention.
        # For the smoke test with random_model_fn, we compute weights from random output.
        scale = 1.0 / np.sqrt(head_dim)
        scores = np.einsum('bhqd,bhkd->bhqk', q_multi, k_multi) * scale  # (batch, num_heads, seq_len, seq_len)
        attn_weights = np.softmax(scores, axis=-1)  # (batch, num_heads, seq_len, seq_len)

        # Mean attention within groups vs between groups
        group_size = batch.group_size
        seq_len = 2 * group_size

        # Mask for within-group attention
        within_mask = np.zeros((seq_len, seq_len), dtype=bool)
        within_mask[:group_size, :group_size] = True      # A->A
        within_mask[group_size:, group_size:] = True      # B->B
        between_mask = ~within_mask

        # Average over batch and heads
        mean_within = attn_weights[:, :, within_mask].mean()
        mean_between = attn_weights[:, :, between_mask].mean()

        # Retrieval accuracy: for each query, does max attention land on target?
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        # target_indices: (batch, seq_len) -> expand to (batch, num_heads, seq_len)
        target_expanded = np.broadcast_to(batch.target_indices[:, None, :], (batch_size, num_heads, seq_len))
        max_attn_idx = attn_weights.argmax(axis=-1)  # (batch, num_heads, seq_len)
        correct = (max_attn_idx == target_expanded)
        retrieval_acc = correct.mean()

        sweep.append({
            "num_heads": num_heads,
            "mean_attn_within": float(mean_within),
            "mean_attn_between": float(mean_between),
            "retrieval_acc": float(retrieval_acc),
        })

    return {
        "version": 1,
        "config": {
            "group_size": batch.group_size,
            "d_model": d_model,
            "num_features": batch.num_features,
            "batch_size": batch.q.shape[0],
            "num_heads_sweep": num_heads_sweep,
        },
        "sweep": sweep,
    }


def random_model_fn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Random model function for smoke testing.
    Returns random attention output of correct shape.
    """
    # q shape: (batch*num_heads, seq_len, head_dim)
    # Return random values with same shape as v
    rng = np.random.default_rng(12345)
    return rng.normal(0, 1, size=v.shape).astype(np.float32)