"""Synthetic induction-head task.

We build sequences that contain a duplicated random "pattern" block:

    [ ... background ... ][ A_0 A_1 ... A_{P-1} ][ A_0 A_1 ... A_{P-1} ][ ... background ... ]

At a second-copy position holding token A_j, a correct induction mechanism
predicts the token that followed the *first* occurrence of A_j, i.e. A_{j+1}.
The separation between the two occurrences of A_j is exactly P (the pattern
length), which is the swept "distance" axis. Larger distance = harder, because
the head must copy from further back.

To keep ground truth unambiguous:
  - pattern tokens are drawn WITHOUT replacement from [0, PATTERN_VOCAB), so a
    pattern token's previous occurrence inside the copy is unique;
  - background tokens are drawn from [PATTERN_VOCAB, VOCAB_SIZE), a disjoint
    range, so background never collides with pattern tokens.

The model function maps token ids to next-token logits:

    model_fn(input_ids: np.ndarray) -> np.ndarray
        input_ids: (batch, seq_len) int
        returns:   (batch, seq_len, vocab_size) float  — next-token logits

`task.evaluate` reads logits at induction target positions, applies softmax,
and reports per-distance accuracy / cross-entropy. Attempts never build the
payload themselves: they hand `evaluate` a model_fn and receive a payload
ready for `benchmark.score`.
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np

ModelFn = Callable[[np.ndarray], np.ndarray]

# ---- Canonical configuration (must match README.md) ----
VOCAB_SIZE = 128            # total token vocabulary
PATTERN_VOCAB = 64          # pattern tokens are [0, PATTERN_VOCAB); background is the rest
SEQ_LEN = 192               # sequence length
DISTANCES = [16, 32, 48, 64]  # pattern length P per bucket == occurrence separation
SEQS_PER_BUCKET = 16        # sequences per distance bucket
BATCH_SIZE = SEQS_PER_BUCKET * len(DISTANCES)  # 64
CANONICAL_DISTANCE = 16     # the canonical (easiest) measurement condition
MAX_TARGETS = max(DISTANCES) - 1  # targets per sequence in the largest bucket
SEED = 42                   # fixed canonical seed


@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray         # (BATCH_SIZE, SEQ_LEN) int32
    target_positions: np.ndarray  # (BATCH_SIZE, MAX_TARGETS) int32; -1 = padding
    target_tokens: np.ndarray     # (BATCH_SIZE, MAX_TARGETS) int32; -1 = padding
    target_distances: np.ndarray  # (BATCH_SIZE, MAX_TARGETS) int32; -1 = padding


def generate(seed: int = SEED) -> Batch:
    """Deterministic canonical batch.

    The `seed` argument is accepted for API compatibility but the canonical
    condition always uses the fixed SEED so that every attempt at this goal
    evaluates on identical data. (Same seed -> same batch regardless.)
    """
    rng = np.random.default_rng(SEED)

    input_ids = np.empty((BATCH_SIZE, SEQ_LEN), dtype=np.int32)
    target_positions = np.full((BATCH_SIZE, MAX_TARGETS), -1, dtype=np.int32)
    target_tokens = np.full((BATCH_SIZE, MAX_TARGETS), -1, dtype=np.int32)
    target_distances = np.full((BATCH_SIZE, MAX_TARGETS), -1, dtype=np.int32)

    row = 0
    for P in DISTANCES:
        for _ in range(SEQS_PER_BUCKET):
            # Background fill from the disjoint background range.
            seq = rng.integers(PATTERN_VOCAB, VOCAB_SIZE, size=SEQ_LEN, dtype=np.int32)

            # Distinct pattern tokens, duplicated contiguously at a random offset.
            A = rng.choice(PATTERN_VOCAB, size=P, replace=False).astype(np.int32)
            max_offset = SEQ_LEN - 2 * P
            offset = int(rng.integers(0, max_offset + 1))
            seq[offset:offset + P] = A
            seq[offset + P:offset + 2 * P] = A

            # Induction targets: at position (offset + P + j) holding A[j],
            # predict A[j+1], for j in 0..P-2.
            n = P - 1
            j = np.arange(n, dtype=np.int32)
            target_positions[row, :n] = offset + P + j
            target_tokens[row, :n] = A[1:1 + n]
            target_distances[row, :n] = P

            input_ids[row] = seq
            row += 1

    return Batch(
        input_ids=input_ids,
        target_positions=target_positions,
        target_tokens=target_tokens,
        target_distances=target_distances,
    )


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch and return the scoring payload."""
    batch = generate()

    logits = np.asarray(model_fn(batch.input_ids), dtype=np.float64)
    expected = (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    if logits.shape != expected:
        raise ValueError(
            f"model_fn returned logits of shape {logits.shape}, expected {expected}"
        )

    valid = batch.target_positions >= 0          # (B, MAX_TARGETS)
    rows = np.where(valid)[0]                     # (N,)
    pos = batch.target_positions[valid]           # (N,)
    tok = batch.target_tokens[valid]              # (N,)
    dist = batch.target_distances[valid]          # (N,)
    if rows.size == 0:
        raise ValueError("No valid induction targets were generated")

    # Logits at each target position: (N, vocab_size)
    tgt_logits = logits[rows, pos, :]

    # Accuracy.
    pred = tgt_logits.argmax(axis=-1)
    correct = (pred == tok).astype(np.float64)

    # Cross-entropy via stable log-softmax.
    shifted = tgt_logits - tgt_logits.max(axis=-1, keepdims=True)
    log_norm = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    log_probs = shifted - log_norm
    ce = -log_probs[np.arange(tok.size), tok]

    uniform_acc = 1.0 / VOCAB_SIZE
    uniform_ce = float(np.log(VOCAB_SIZE))

    sweep = []
    for P in DISTANCES:
        mask = dist == P
        cnt = int(mask.sum())
        if cnt == 0:
            acc = 0.0
            loss = 0.0
        else:
            acc = float(correct[mask].mean())
            loss = float(ce[mask].mean())
        sweep.append({
            "distance": int(P),
            "num_targets": cnt,
            "accuracy": acc,
            "ce_loss": loss,
            "uniform_baseline_accuracy": uniform_acc,
            "uniform_baseline_ce_loss": uniform_ce,
        })

    return {
        "version": 1,
        "model_name": "synthetic_induction",
        "vocab_size": VOCAB_SIZE,
        "seq_len": SEQ_LEN,
        "canonical_distance": CANONICAL_DISTANCE,
        "sweep": sweep,
        "aggregate": {
            "accuracy": float(correct.mean()),
            "ce_loss": float(ce.mean()),
            "num_targets": int(correct.size),
            "uniform_baseline_accuracy": uniform_acc,
            "uniform_baseline_ce_loss": uniform_ce,
        },
    }


def random_model_fn() -> ModelFn:
    """Return a model_fn producing uniform (zero) logits — same signature as a real one.

    Pure NumPy, no torch / GPU. Used by the pipeline smoke test:
        payload = task.evaluate(task.random_model_fn())
    """
    def _fn(input_ids: np.ndarray) -> np.ndarray:
        b, s = np.asarray(input_ids).shape
        return np.zeros((b, s, VOCAB_SIZE), dtype=np.float32)

    return _fn


if __name__ == "__main__":
    import json
    payload = evaluate(random_model_fn())
    print(json.dumps({k: v for k, v in payload.items() if k != "sweep"}, indent=2))
    print("sweep buckets:", len(payload["sweep"]))
