"""
Task definition for `attention_int_add`.

The goal asks: how well does an attention-based model implement multi-digit
integer addition, and — crucially — how robust is that mechanism to *carry
propagation*, the part of addition that requires routing information between
digit columns?

Exports
-------
generate(seed=0) -> Batch
    Deterministic synthetic addition problems, bucketed by number of carries.
evaluate(model_fn) -> dict
    Runs `model_fn` over the batch and returns the payload that
    `benchmark.score` consumes.
random_model_fn() -> ModelFn
    A dummy `model_fn` (pure NumPy) used by the smoke test.

model_fn contract
-----------------
    model_fn(input_ids: np.ndarray[int]  shape (N, SEQ_LEN))
        -> logits: np.ndarray[float] shape (N, SEQ_LEN, VOCAB_SIZE)

The model is shown the problem with the answer region masked to PAD_TOKEN and
must predict the sum digits at the SUM positions. `evaluate` reads the argmax
of `logits` at the SUM positions only.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

ModelFn = Callable[[np.ndarray], np.ndarray]

# ----------------------------------------------------------------------
# Vocabulary & sequence layout
# ----------------------------------------------------------------------
# Token IDs 0-9 are the literal digits.
PLUS_TOKEN = 10
EQUALS_TOKEN = 11
BOS_TOKEN = 12
EOS_TOKEN = 13
PAD_TOKEN = 14
VOCAB_SIZE = 15

MAX_DIGITS = 3                       # operands are <= 3 digits (0..999)
SUM_DIGITS = MAX_DIGITS + 1          # sum of two 3-digit numbers can be 4 digits

# Layout (MAX_DIGITS=3, most-significant digit first):
#   idx 0          : BOS
#   idx 1..3       : operand A digits (zero-padded to MAX_DIGITS)
#   idx 4          : PLUS
#   idx 5..7       : operand B digits
#   idx 8          : EQUALS
#   idx 9..12      : SUM digits (zero-padded to SUM_DIGITS) -- masked on input
#   idx 13         : EOS
SUM_START_IDX = 2 * MAX_DIGITS + 3   # = 9
SEQ_LEN = SUM_START_IDX + SUM_DIGITS + 1  # = 14
SUM_POSITIONS = list(range(SUM_START_IDX, SUM_START_IDX + SUM_DIGITS))

# Sweep axis: exact number of carries in the column-wise addition (0..MAX_DIGITS).
CARRY_SWEEP: Tuple[int, ...] = tuple(range(MAX_DIGITS + 1))  # (0, 1, 2, 3)
CANONICAL_CARRIES = MAX_DIGITS        # the hardest condition: full carry chain

NUM_SAMPLES_PER_SLICE = 300
_POOL_SIZE = 400_000                  # rejection pool to fill every carry bucket


# ----------------------------------------------------------------------
# Batch
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Batch:
    input_ids: np.ndarray                 # (N, SEQ_LEN) int32, SUM region masked to PAD
    target_sum_digits: np.ndarray         # (N, SUM_DIGITS) int32, MSB first
    num_carries: np.ndarray               # (N,) int32
    operand_a: np.ndarray                 # (N,) int32
    operand_b: np.ndarray                 # (N,) int32
    slice_indices: List[Tuple[int, int]]  # (start, end) per CARRY_SWEEP value
    carry_sweep: Tuple[int, ...]          # echo of CARRY_SWEEP, in slice order


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _digits_msb(values: np.ndarray, n_digits: int) -> np.ndarray:
    """Return (len(values), n_digits) array of base-10 digits, MSB first."""
    values = np.asarray(values, dtype=np.int64)
    out = np.zeros((values.shape[0], n_digits), dtype=np.int32)
    for i in range(n_digits):
        power = 10 ** (n_digits - 1 - i)
        out[:, i] = (values // power) % 10
    return out


def _count_carries(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised count of carries when adding a + b column by column."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    carry = np.zeros(a.shape[0], dtype=np.int64)
    num_carries = np.zeros(a.shape[0], dtype=np.int64)
    for col in range(MAX_DIGITS):
        power = 10 ** col
        da = (a // power) % 10
        db = (b // power) % 10
        total = da + db + carry
        carry = (total >= 10).astype(np.int64)
        num_carries += carry
    return num_carries.astype(np.int32)


# ----------------------------------------------------------------------
# generate
# ----------------------------------------------------------------------
def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed: same seed -> identical Batch."""
    rng = np.random.default_rng(seed)

    a_pool = rng.integers(0, 10 ** MAX_DIGITS, size=_POOL_SIZE, dtype=np.int64)
    b_pool = rng.integers(0, 10 ** MAX_DIGITS, size=_POOL_SIZE, dtype=np.int64)
    carries_pool = _count_carries(a_pool, b_pool)

    a_list: List[int] = []
    b_list: List[int] = []
    carries_list: List[int] = []
    slice_indices: List[Tuple[int, int]] = []

    cursor = 0
    for k in CARRY_SWEEP:
        idx = np.nonzero(carries_pool == k)[0][:NUM_SAMPLES_PER_SLICE]
        if idx.size == 0:
            raise ValueError(
                f"carry bucket k={k} unfilled; increase _POOL_SIZE (seed={seed})"
            )
        a_list.extend(a_pool[idx].tolist())
        b_list.extend(b_pool[idx].tolist())
        carries_list.extend([k] * idx.size)
        slice_indices.append((cursor, cursor + idx.size))
        cursor += idx.size

    a = np.asarray(a_list, dtype=np.int64)
    b = np.asarray(b_list, dtype=np.int64)
    s = a + b
    n = a.shape[0]

    a_digits = _digits_msb(a, MAX_DIGITS)          # (n, 3)
    b_digits = _digits_msb(b, MAX_DIGITS)          # (n, 3)
    sum_digits = _digits_msb(s, SUM_DIGITS)        # (n, 4)

    input_ids = np.full((n, SEQ_LEN), PAD_TOKEN, dtype=np.int32)
    input_ids[:, 0] = BOS_TOKEN
    input_ids[:, 1:1 + MAX_DIGITS] = a_digits
    input_ids[:, 1 + MAX_DIGITS] = PLUS_TOKEN
    input_ids[:, 2 + MAX_DIGITS:2 + 2 * MAX_DIGITS] = b_digits
    input_ids[:, 2 + 2 * MAX_DIGITS] = EQUALS_TOKEN
    # SUM positions left as PAD_TOKEN (masked); model must predict them.
    input_ids[:, SUM_START_IDX + SUM_DIGITS] = EOS_TOKEN

    return Batch(
        input_ids=input_ids,
        target_sum_digits=sum_digits.astype(np.int32),
        num_carries=np.asarray(carries_list, dtype=np.int32),
        operand_a=a.astype(np.int32),
        operand_b=b.astype(np.int32),
        slice_indices=slice_indices,
        carry_sweep=CARRY_SWEEP,
    )


# ----------------------------------------------------------------------
# Linear (no-carry) baseline -- model-independent, derived from the data
# ----------------------------------------------------------------------
def _linear_baseline_prediction(batch: Batch) -> np.ndarray:
    """
    The strawman that adds each column independently mod 10 and ignores all
    carries (leading sum digit predicted as 0). Returns (N, SUM_DIGITS).
    """
    a_digits = _digits_msb(batch.operand_a, MAX_DIGITS)
    b_digits = _digits_msb(batch.operand_b, MAX_DIGITS)
    col_sum = (a_digits + b_digits) % 10            # (n, 3)
    pred = np.zeros((batch.operand_a.shape[0], SUM_DIGITS), dtype=np.int32)
    pred[:, SUM_DIGITS - MAX_DIGITS:] = col_sum     # align to least-significant cols
    # leading digit stays 0
    return pred


def _slice_stats(pred: np.ndarray, target: np.ndarray,
                 start: int, end: int) -> Tuple[float, float, int]:
    """Return (exact_match_rate, digit_accuracy, n) over a [start, end) slice."""
    if end <= start:
        return 0.0, 0.0, 0
    p = pred[start:end]
    t = target[start:end]
    n = int(end - start)
    correct_digits = (p == t)
    digit_accuracy = float(correct_digits.mean())
    exact_match = float(correct_digits.all(axis=1).mean())
    return exact_match, digit_accuracy, n


# ----------------------------------------------------------------------
# evaluate
# ----------------------------------------------------------------------
def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload."""
    batch = generate(seed=0)

    logits = np.asarray(model_fn(batch.input_ids), dtype=np.float64)
    expected_shape = (batch.input_ids.shape[0], SEQ_LEN, VOCAB_SIZE)
    if logits.shape != expected_shape:
        raise ValueError(
            f"model_fn must return logits of shape {expected_shape}, "
            f"got {logits.shape}"
        )

    # Predicted digits = argmax over vocab at SUM positions.
    sum_logits = logits[:, SUM_POSITIONS, :]                       # (N, SUM_DIGITS, VOCAB)
    pred_digits = np.argmax(sum_logits, axis=-1).astype(np.int32)  # (N, SUM_DIGITS)

    base_pred = _linear_baseline_prediction(batch)
    target = batch.target_sum_digits

    sweep = []
    linear_baseline = []
    for k, (start, end) in zip(batch.carry_sweep, batch.slice_indices):
        em, dacc, n = _slice_stats(pred_digits, target, start, end)
        sweep.append({
            "carries": int(k),
            "exact_match_rate": em,
            "digit_accuracy": dacc,
            "n": n,
        })
        b_em, b_dacc, _ = _slice_stats(base_pred, target, start, end)
        linear_baseline.append({
            "carries": int(k),
            "exact_match_rate": b_em,
            "digit_accuracy": b_dacc,
            "n": n,
        })

    return {
        "version": 1,
        "max_digits": MAX_DIGITS,
        "sum_digits": SUM_DIGITS,
        "canonical_carries": CANONICAL_CARRIES,
        "carry_sweep": [int(k) for k in batch.carry_sweep],
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


# ----------------------------------------------------------------------
# random_model_fn -- smoke-test dummy
# ----------------------------------------------------------------------
def random_model_fn() -> ModelFn:
    """A pure-NumPy dummy with the real model_fn signature: random logits."""
    rng = np.random.default_rng(12345)

    def _fn(input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids)
        n = input_ids.shape[0]
        return rng.standard_normal((n, SEQ_LEN, VOCAB_SIZE))

    return _fn
