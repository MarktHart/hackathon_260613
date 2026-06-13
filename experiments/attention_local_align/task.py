from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray          # (B, T) int32
    target_indices: np.ndarray     # (B, T) int32; true target position per query, -1 if none


ModelFn = Callable[[np.ndarray], np.ndarray]
# input:  (B, T) int32 token ids
# output: (B, H, T, T) float32 attention weights, softmax-normalised per query (last dim sums to 1)

VOCAB_SIZE = 64
SEQ_LEN = 32
BATCH_SIZE = 128
MEASURED_HEAD = 0
CANONICAL_SHIFT = -1
SHIFTS = [-2, -1, 0, 1, 2]


def _make_vocab(seed: int, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**31 - 1, size=vocab_size, dtype=np.int32)


def _generate_batch_for_shift(
    shift: int,
    batch_size: int,
    seq_len: int,
    vocab: np.ndarray,
    seed: int,
) -> Batch:
    """Generate a batch where each query attends to query_pos + shift (clamped)."""
    rng = np.random.default_rng(seed + (shift + 1000) * 1009)  # deterministic per shift
    input_ids = rng.choice(vocab, size=(batch_size, seq_len), replace=True).astype(np.int32)

    t = np.arange(seq_len, dtype=np.int32)
    target = t + shift
    target = np.where((target >= 0) & (target < seq_len), target, -1)
    target_indices = np.tile(target, (batch_size, 1)).astype(np.int32)

    return Batch(input_ids=input_ids, target_indices=target_indices)


def generate(seed: int = 0) -> Batch:
    """
    Generate the canonical batch (shift = -1).
    Deterministic: same seed → same batch. `seed` only reshuffles the vocab.
    """
    vocab = _make_vocab(seed)
    return _generate_batch_for_shift(
        shift=CANONICAL_SHIFT, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, vocab=vocab, seed=seed
    )


def _evaluate_one_shift(
    model_fn: ModelFn,
    shift: int,
    batch_size: int,
    seq_len: int,
    vocab: np.ndarray,
    seed: int,
) -> dict:
    batch = _generate_batch_for_shift(shift, batch_size, seq_len, vocab, seed)

    attn = np.asarray(model_fn(batch.input_ids), dtype=np.float64)  # (B, H, T, T)
    if attn.ndim != 4:
        raise ValueError(
            f"model_fn must return a 4-D (B, H, T, T) array, got ndim={attn.ndim}"
        )
    B, H, Tq, Tk = attn.shape
    if H <= MEASURED_HEAD:
        raise ValueError(
            f"model_fn returned {H} heads, need at least {MEASURED_HEAD + 1}"
        )
    if (B, Tq, Tk) != (batch_size, seq_len, seq_len):
        raise ValueError(
            f"model_fn returned shape {attn.shape}, expected "
            f"({batch_size}, H, {seq_len}, {seq_len})"
        )

    attn_h = attn[:, MEASURED_HEAD, :, :]  # (B, T, T)
    target = batch.target_indices          # (B, T)
    valid = target != -1                   # (B, T)

    max_attn_to_target = np.zeros((B, seq_len), dtype=np.float64)
    peak_on_target = np.zeros((B, seq_len), dtype=np.bool_)
    entropy = np.zeros((B, seq_len), dtype=np.float64)

    argmax = np.argmax(attn_h, axis=2)  # (B, T)
    safe_target = np.where(valid, target, 0)
    gathered = np.take_along_axis(attn_h, safe_target[:, :, None], axis=2)[:, :, 0]
    max_attn_to_target = np.where(valid, gathered, 0.0)
    peak_on_target = valid & (argmax == target)
    p = np.clip(attn_h, 1e-12, 1.0)
    entropy = -np.sum(p * np.log(p), axis=2)  # (B, T)

    valid_count = int(np.sum(valid))
    if valid_count == 0:
        return {
            "shift": int(shift),
            "mean_max_attn_to_target": 0.0,
            "mean_entropy": 0.0,
            "frac_peak_on_target": 0.0,
        }

    return {
        "shift": int(shift),
        "mean_max_attn_to_target": float(np.sum(max_attn_to_target[valid]) / valid_count),
        "mean_entropy": float(np.sum(entropy[valid]) / valid_count),
        "frac_peak_on_target": float(np.sum(peak_on_target[valid]) / valid_count),
    }


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run model_fn over the canonical sweep of shifts and return the payload
    matching benchmark.score's contract exactly.
    """
    seed = 0  # fixed seed for canonical evaluation
    vocab = _make_vocab(seed)

    sweep = [
        _evaluate_one_shift(model_fn, shift, BATCH_SIZE, SEQ_LEN, vocab, seed)
        for shift in SHIFTS
    ]

    return {
        "version": 1,
        "canonical_shift": CANONICAL_SHIFT,
        "sequence_length": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "batch_size": BATCH_SIZE,
        "measured_head": MEASURED_HEAD,
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """
    Return a ModelFn whose body emits random (per-query softmax-normalised)
    attention weights of the right shape. Pure NumPy, no torch.
    """
    rng = np.random.default_rng(0)

    def _fn(input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids)
        B, T = input_ids.shape
        H = 4  # arbitrary number of heads
        logits = rng.normal(size=(B, H, T, T))
        logits = logits - logits.max(axis=3, keepdims=True)
        w = np.exp(logits)
        w = w / w.sum(axis=3, keepdims=True)
        return w.astype(np.float32)

    return _fn
