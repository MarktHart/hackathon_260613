from dataclasses import dataclass
import numpy as np
from typing import Protocol

# ──────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Batch:
    sequences: np.ndarray          # (n_sequences, seq_len), token IDs
    anchor_pos: int                # always 0
    wildcard_pos: int              # always 1 (first wildcard)
    target_pos: int                # 1 + wildcard_span + 1
    wildcard_span: int             # number of wildcard tokens (0..4)
    anchor_token: int
    target_token: int
    wildcard_token_range: tuple[int, int]  # inclusive [low, high]


# ModelFn: takes a Batch, returns attention weights (n_sequences, seq_len, seq_len)
class ModelFn(Protocol):
    def __call__(self, batch: Batch) -> np.ndarray: ...


# ──────────────────────────────────────────────────────────────
# Constants (canonical condition)
# ──────────────────────────────────────────────────────────────

CANONICAL_SPAN = 1
N_SEQUENCES = 1024
SEQ_LEN = 16
VOCAB_SIZE = 32
ANCHOR_TOKEN = 1
TARGET_TOKEN = 2
WILDCARD_RANGE = (10, 31)          # inclusive
SWEEP_SPANS = (0, 1, 2, 3, 4)

# ──────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────

def _make_batch_for_span(wildcard_span: int, seed: int) -> Batch:
    """Deterministic batch for a single wildcard_span."""
    rng = np.random.default_rng(seed + wildcard_span * 1000)
    n_seq = N_SEQUENCES
    seq_len = SEQ_LEN

    sequences = np.zeros((n_seq, seq_len), dtype=np.int32)

    anchor_pos = 0
    wildcard_pos = 1
    target_pos = 1 + wildcard_span  # anchor(0) + wildcards(1..k) -> target at k+1

    # Anchor token at position 0
    sequences[:, anchor_pos] = ANCHOR_TOKEN

    # Wildcard positions 1 .. 1+k-1 filled with random distractor tokens
    if wildcard_span > 0:
        low, high = WILDCARD_RANGE
        sequences[:, wildcard_pos:wildcard_pos + wildcard_span] = rng.integers(
            low, high + 1, size=(n_seq, wildcard_span)
        )

    # Target token at target_pos
    sequences[:, target_pos] = TARGET_TOKEN

    # Filler positions (everything else) stay 0 (padding token).

    return Batch(
        sequences=sequences,
        anchor_pos=anchor_pos,
        wildcard_pos=wildcard_pos,
        target_pos=target_pos,
        wildcard_span=wildcard_span,
        anchor_token=ANCHOR_TOKEN,
        target_token=TARGET_TOKEN,
        wildcard_token_range=WILDCARD_RANGE,
    )


def generate(seed: int = 0) -> Batch:
    """
    Generate a Batch for the *canonical* wildcard_span (1).
    The seed controls the random wildcard tokens; same seed → same batch.
    """
    return _make_batch_for_span(CANONICAL_SPAN, seed)


# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────

def evaluate(model_fn: ModelFn) -> dict:
    """
    Run model_fn on batches for all wildcard_spans in SWEEP_SPANS,
    collect attention stats, return a payload dict that benchmark.score consumes.
    """
    sweep_records = []

    for span in SWEEP_SPANS:
        # Fixed seed per span so results are deterministic across attempts.
        batch = _make_batch_for_span(span, seed=42)

        attn = np.asarray(model_fn(batch), dtype=np.float64)  # (N, L, L)
        if attn.shape != (N_SEQUENCES, SEQ_LEN, SEQ_LEN):
            raise ValueError(
                f"model_fn returned shape {attn.shape}, expected "
                f"({N_SEQUENCES}, {SEQ_LEN}, {SEQ_LEN})"
            )

        # Target's attention row: query = target_pos → (n_sequences, seq_len)
        target_attn = attn[:, batch.target_pos, :]

        # Mean attention on anchor (position 0)
        mean_attn_on_anchor = float(target_attn[:, batch.anchor_pos].mean())

        # Mean attention on wildcard positions
        if batch.wildcard_span > 0:
            wc_slice = slice(batch.wildcard_pos, batch.wildcard_pos + batch.wildcard_span)
            mean_attn_on_wildcards = float(target_attn[:, wc_slice].mean())
        else:
            mean_attn_on_wildcards = 0.0

        # Mean attention on all other positions (excl. anchor, wildcards, target)
        mask = np.ones(SEQ_LEN, dtype=bool)
        mask[batch.anchor_pos] = False
        mask[batch.target_pos] = False
        if batch.wildcard_span > 0:
            mask[batch.wildcard_pos:batch.wildcard_pos + batch.wildcard_span] = False
        mean_attn_on_others = float(target_attn[:, mask].mean()) if mask.any() else 0.0

        # Sharpness = anchor / (wildcards + others + eps)
        denom = mean_attn_on_wildcards + mean_attn_on_others + 1e-8
        sharpness = mean_attn_on_anchor / denom

        sweep_records.append({
            "wildcard_span": span,
            "wildcard_pos": batch.wildcard_pos,
            "target_pos": batch.target_pos,
            "n_sequences": N_SEQUENCES,
            "mean_attn_on_anchor": mean_attn_on_anchor,
            "mean_attn_on_wildcards": mean_attn_on_wildcards,
            "mean_attn_on_others": mean_attn_on_others,
            "sharpness": float(sharpness),
        })

    payload = {
        "version": 1,
        "canonical_span": CANONICAL_SPAN,
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "anchor_token": ANCHOR_TOKEN,
        "target_token": TARGET_TOKEN,
        "wildcard_token_range": list(WILDCARD_RANGE),
        "sweep": sweep_records,
    }
    return payload


# ──────────────────────────────────────────────────────────────
# Random model function (smoke test)
# ──────────────────────────────────────────────────────────────

def random_model_fn() -> ModelFn:
    """
    Returns a ModelFn that outputs uniform attention (1/seq_len everywhere).
    Pure NumPy, no torch, no GPU. Matches the real model_fn signature exactly.
    """
    def _fn(batch: Batch) -> np.ndarray:
        n_seq = batch.sequences.shape[0]
        seq_len = batch.sequences.shape[1]
        return np.full((n_seq, seq_len, seq_len), 1.0 / seq_len, dtype=np.float32)
    return _fn
