"""Synthetic FSM state-tracking task: predict DFA state from token sequences.

Exports:
    generate(seed) -> Batch
    evaluate(model_fn) -> payload dict   (consumed by benchmark.score)
    random_model_fn() -> model_fn        (reference chance-level model)
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Batch:
    tokens: np.ndarray       # [num_sequences, seq_len] int32, values 0..3
    true_states: np.ndarray  # [num_sequences, seq_len] int32, values 0..2


# ---- DFA specification (matches README) ----
_NUM_STATES = 3
_ALPHABET_SIZE = 4
_TRANSITION = np.array(
    [
        [0, 1, 2, 1],  # from state 0
        [1, 2, 0, 2],  # from state 1
        [2, 0, 1, 0],  # from state 2
    ],
    dtype=np.int32,
)  # delta[s][t] = next_state

# ---- canonical measurement condition ----
_BURNIN = 16
_SEQ_LEN = 64
_NUM_SEQUENCES = 128


def _generate_sequences(seed: int) -> Batch:
    rng = np.random.default_rng(seed)
    tokens = rng.integers(
        0, _ALPHABET_SIZE, size=(_NUM_SEQUENCES, _SEQ_LEN), dtype=np.int32
    )
    true_states = np.zeros((_NUM_SEQUENCES, _SEQ_LEN), dtype=np.int32)
    true_states[:, 0] = rng.integers(0, _NUM_STATES, size=_NUM_SEQUENCES, dtype=np.int32)
    for t in range(1, _SEQ_LEN):
        prev = true_states[:, t - 1]
        tok = tokens[:, t]
        true_states[:, t] = _TRANSITION[prev, tok]
    return Batch(tokens=tokens, true_states=true_states)


def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed."""
    return _generate_sequences(seed)


def evaluate(model_fn) -> dict:
    """Run model_fn over the canonical batch (seed 0); return the payload dict.

    model_fn(tokens) -> logits float array [num_sequences, seq_len, num_states].
    """
    batch = _generate_sequences(0)
    logits = np.asarray(model_fn(batch.tokens))

    expected = (_NUM_SEQUENCES, _SEQ_LEN, _NUM_STATES)
    if logits.shape != expected:
        raise ValueError(
            f"model_fn returned shape {logits.shape}, expected {expected}"
        )

    preds = logits.argmax(axis=-1).astype(np.int32)  # [batch, seq_len]
    correct = (preds == batch.true_states).astype(np.float64)

    per_position_accuracy = correct.mean(axis=0).tolist()  # length seq_len
    overall_accuracy = float(correct[:, _BURNIN:].mean())

    # post-burnin slices
    true_pb = batch.true_states[:, _BURNIN:]
    pred_pb = preds[:, _BURNIN:]

    per_state_recall = []
    for s in range(_NUM_STATES):
        mask = true_pb == s
        denom = int(mask.sum())
        recall = float((pred_pb[mask] == s).mean()) if denom > 0 else 0.0
        per_state_recall.append(recall)

    confusion = np.zeros((_NUM_STATES, _NUM_STATES), dtype=np.int64)
    for true_s in range(_NUM_STATES):
        for pred_s in range(_NUM_STATES):
            confusion[true_s, pred_s] = int(
                ((true_pb == true_s) & (pred_pb == pred_s)).sum()
            )

    return {
        "version": 1,
        "seq_len": _SEQ_LEN,
        "num_sequences": _NUM_SEQUENCES,
        "burnin": _BURNIN,
        "dfa_spec": {
            "num_states": _NUM_STATES,
            "alphabet_size": _ALPHABET_SIZE,
            "transition": _TRANSITION.tolist(),
            "token_map": {"A": 0, "B": 1, "C": 2, "D": 3},
        },
        "per_position_accuracy": per_position_accuracy,
        "overall_accuracy": overall_accuracy,
        "random_baseline_accuracy": 1.0 / _NUM_STATES,
        "per_state_recall": per_state_recall,
        "transition_confusion": confusion.tolist(),
    }


def random_model_fn():
    """Return a reference model_fn emitting constant (uniform) state logits.

    Signature matches a real model_fn: tokens -> logits
    [num_sequences, seq_len, num_states]. Pure NumPy, chance-level tracking.
    """

    def model_fn(tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens)
        batch, seq_len = tokens.shape
        return np.zeros((batch, seq_len, _NUM_STATES), dtype=np.float32)

    return model_fn
