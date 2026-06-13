import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, Any

# model_fn signature (the goal's contract with attempts):
#   model_fn(input_ids: np.ndarray[int32, (batch, seq_len)], delim_id: int)
#       -> np.ndarray[float, (batch, n_heads, seq_len, seq_len)]
# The returned array is per-head attention weights; the last axis (keys) must
# be a probability distribution (sums to 1).
ModelFn = Callable[[np.ndarray, int], np.ndarray]


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray          # (batch, seq_len) int32
    delim_id: int                  # scalar
    seg_len: int                   # scalar
    delim_pos: int                 # scalar
    n_heads: int                   # scalar (also fixes random_model_fn shape)


def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch generation. Same seed -> same batch.

    Structure is fixed (segA, DELIM, segB, EOS); `seed` only affects which
    tokens are sampled within each segment's vocab class.
    """
    rng = np.random.default_rng(seed)

    # Fixed canonical config
    vocab_size = 64
    seg_len = 8
    batch_size = 32
    n_heads = 4
    delim_id = vocab_size - 1      # 63
    eos_id = vocab_size - 2        # 62
    segA_vocab = np.arange(1, vocab_size // 2, dtype=np.int32)               # 1..31
    segB_vocab = np.arange(vocab_size // 2, vocab_size - 2, dtype=np.int32)  # 32..61

    delim_pos = seg_len                         # index 8
    seq_len = 2 * seg_len + 2                    # 18 (segA[8], DELIM[1], segB[8], EOS[1])

    input_ids = np.zeros((batch_size, seq_len), dtype=np.int32)
    for b in range(batch_size):
        input_ids[b, :seg_len] = rng.choice(segA_vocab, size=seg_len, replace=True)
        input_ids[b, delim_pos] = delim_id
        input_ids[b, delim_pos + 1:delim_pos + 1 + seg_len] = rng.choice(segB_vocab, size=seg_len, replace=True)
        input_ids[b, -1] = eos_id

    return Batch(
        input_ids=input_ids,
        delim_id=int(delim_id),
        seg_len=seg_len,
        delim_pos=delim_pos,
        n_heads=n_heads,
    )


def _region_metrics(q_attn: np.ndarray,
                    within_key_start: int, within_key_end: int,
                    cross_key_start: int, cross_key_end: int,
                    delim_pos: int) -> Dict[str, Any]:
    """
    q_attn: (batch, n_heads, q_len, seq_len) attention weights for one set of
    query positions.

    All region quantities are the total attention MASS to that region
    (summed over the region's key positions), then averaged over batch, heads
    and query positions. The four regions partition all keys, so
    within + delim + cross + eos == 1 (up to float error).
    """
    # Mass to each region: sum over the relevant key positions -> (batch, n_heads, q_len)
    within_mass = q_attn[:, :, :, within_key_start:within_key_end].sum(axis=-1)
    delim_mass = q_attn[:, :, :, delim_pos]
    cross_mass = q_attn[:, :, :, cross_key_start:cross_key_end].sum(axis=-1)
    eos_mass = q_attn[:, :, :, -1]

    # Scalar means over batch + query positions + heads
    within = float(within_mass.mean())
    delim = float(delim_mass.mean())
    cross = float(cross_mass.mean())
    eos = float(eos_mass.mean())

    # Per-head means over batch + query positions -> (n_heads,)
    head_within = within_mass.mean(axis=(0, 2))
    head_delim = delim_mass.mean(axis=(0, 2))
    head_cross = cross_mass.mean(axis=(0, 2))
    head_eos = eos_mass.mean(axis=(0, 2))

    # Per-head sharpness: how much more mass goes within-segment than to the
    # best competing region (delimiter, cross-segment, or EOS).
    head_sharpness = (
        head_within - np.maximum(np.maximum(head_delim, head_cross), head_eos)
    ).astype(float).tolist()

    return {
        "within_seg_attn": within,
        "delim_attn": delim,
        "cross_seg_attn": cross,
        "eos_attn": eos,
        "head_sharpness": head_sharpness,
    }


def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """
    Run `model_fn` on the canonical batch and return the payload dict exactly
    as benchmark.score() consumes it.
    """
    batch = generate(seed=0)  # canonical seed

    attn = np.asarray(model_fn(batch.input_ids, batch.delim_id))

    batch_size, seq_len = batch.input_ids.shape
    seg_len = batch.seg_len
    delim_pos = batch.delim_pos
    n_heads = batch.n_heads

    expected_shape = (batch_size, n_heads, seq_len, seq_len)
    if attn.shape != expected_shape:
        raise ValueError(f"model_fn returned shape {attn.shape}, expected {expected_shape}")

    attn = attn.astype(np.float64)
    sums = attn.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        raise ValueError(
            f"Attention weights must sum to 1 over key dim: max|sum-1| = "
            f"{float(np.max(np.abs(sums - 1.0))):.6f}"
        )

    # Segment A queries: [0, seg_len); within=segA, cross=segB
    segA = _region_metrics(
        attn[:, :, 0:seg_len, :],
        within_key_start=0, within_key_end=seg_len,
        cross_key_start=delim_pos + 1, cross_key_end=delim_pos + 1 + seg_len,
        delim_pos=delim_pos,
    )
    # Segment B queries: [delim_pos+1, delim_pos+1+seg_len); within=segB, cross=segA
    segB = _region_metrics(
        attn[:, :, delim_pos + 1:delim_pos + 1 + seg_len, :],
        within_key_start=delim_pos + 1, within_key_end=delim_pos + 1 + seg_len,
        cross_key_start=0, cross_key_end=seg_len,
        delim_pos=delim_pos,
    )

    # Linear (no-mechanism) baseline: uniform attention over all keys, which
    # has no boundary awareness. Region masses are proportional to region size.
    uniform = 1.0 / seq_len
    base_region = {
        "within_seg_attn": float(uniform * seg_len),
        "delim_attn": float(uniform),
        "cross_seg_attn": float(uniform * seg_len),
        "eos_attn": float(uniform),
        "head_sharpness": [0.0] * n_heads,  # within - max(delim, cross, eos) = 0
    }
    linear_baseline = {"segA": dict(base_region), "segB": dict(base_region)}

    payload = {
        "version": 1,
        "config": {
            "vocab_size": 64,
            "seg_len": seg_len,
            "batch_size": batch_size,
            "n_heads": n_heads,
            "seq_len": seq_len,
            "delim_pos": delim_pos,
            "canonical_seed": 0,
        },
        "sweep": [
            {"query_segment": "segA", **segA},
            {"query_segment": "segB", **segB},
        ],
        "linear_baseline": linear_baseline,
    }
    return payload


def random_model_fn() -> ModelFn:
    """
    A callable with the exact same signature as a real model_fn, returning a
    valid (uniform) attention distribution of the right shape. Pure NumPy.
    """
    def _fn(input_ids: np.ndarray, delim_id: int) -> np.ndarray:
        batch, seq_len = np.asarray(input_ids).shape
        n_heads = 4  # fixed by generate()
        return np.full((batch, n_heads, seq_len, seq_len), 1.0 / seq_len, dtype=np.float32)
    return _fn
