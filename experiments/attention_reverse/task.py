import numpy as np
from dataclasses import dataclass
from typing import Callable

# --- model_fn contract -------------------------------------------------------
#
# A model_fn takes a batch of token ids and returns BOTH next-token-style logits
# for the reversal task AND the attention pattern of the head implementing the
# reversal, so the benchmark can measure the mechanism, not just the behaviour.
#
#   model_fn(tokens) -> (logits, attn)
#       tokens : np.ndarray (batch, seq_len)          int   token ids in [0, vocab)
#       logits : np.ndarray (batch, seq_len, vocab)   float prediction at each pos
#       attn   : np.ndarray (batch, seq_len, seq_len) float attention weights;
#                attn[b, i, j] = weight query position i places on key position j.
#                Rows are normalised over j (evaluate re-normalises defensively).
#
# Attention is BIDIRECTIONAL (no causal mask): position i must predict the token
# at the mirror position seq_len-1-i, which for early positions lies in the
# future, so a causal mask would make the task unsolvable. See README.md.
ModelFn = Callable[[np.ndarray], "tuple[np.ndarray, np.ndarray]"]

# --- Canonical measurement condition (see README.md) -------------------------
VOCAB_SIZE = 16
BATCH_SIZE = 32
CANONICAL_SEQ_LEN = 16
CANONICAL_IDX = 1          # index of CANONICAL_SEQ_LEN within SEQ_LEN_SWEEP

# Length-generalisation sweep. Canonical length is 16; 32 and 64 probe whether
# the mechanism extrapolates beyond the canonical length.
SEQ_LEN_SWEEP = [8, 16, 32, 64]
NUM_SEQUENCES = 256        # sequences evaluated per slice
EVAL_SEED = 42


@dataclass(frozen=True)
class Batch:
    """One slice of the reversal task at a fixed sequence length."""
    tokens: np.ndarray            # (n, seq_len) int32
    targets: np.ndarray           # (n, seq_len) int32 — token at the mirror position
    mirror_positions: np.ndarray  # (seq_len,) int32 — mirror index for each position
    seq_len: int


def _make_slice(seq_len: int, num_sequences: int, seed: int) -> Batch:
    """Deterministic reversal data for one sequence length."""
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, VOCAB_SIZE, size=(num_sequences, seq_len), dtype=np.int32)
    mirror_positions = np.arange(seq_len - 1, -1, -1, dtype=np.int32)  # [S-1, ..., 0]
    targets = tokens[:, mirror_positions]
    return Batch(
        tokens=tokens,
        targets=targets,
        mirror_positions=mirror_positions,
        seq_len=seq_len,
    )


def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch for the canonical condition
    (seq_len=16, vocab=16, num_sequences=256). Same seed -> same batch.
    """
    return _make_slice(CANONICAL_SEQ_LEN, NUM_SEQUENCES, seed)


def _normalise_attn(attn, n: int, seq_len: int) -> np.ndarray:
    """Coerce attention to (n, seq_len, seq_len) and row-normalise over keys."""
    attn = np.asarray(attn, dtype=np.float64)
    if attn.shape != (n, seq_len, seq_len):
        raise ValueError(
            f"model_fn attn has shape {attn.shape}, expected ({n}, {seq_len}, {seq_len})"
        )
    attn = np.clip(attn, 0.0, None)
    row_sums = attn.sum(axis=-1, keepdims=True)
    # Degenerate (all-zero) rows are renormalised to uniform so they report
    # ~1/seq_len mirror mass (chance), as documented in README.md, rather than
    # collapsing to 0 mass.
    degenerate = row_sums <= 1e-12
    safe_sums = np.where(degenerate, 1.0, row_sums)
    normalised = attn / safe_sums
    return np.where(degenerate, 1.0 / seq_len, normalised)


def _eval_slice(model_fn: ModelFn, batch: Batch) -> dict:
    """Run model_fn over one slice and return aggregate stats for that slice."""
    seq_len = batch.seq_len
    mirror = batch.mirror_positions  # (seq_len,)

    total = 0
    correct = 0
    attn_mass_sum = 0.0
    attn_mass_count = 0

    for i in range(0, batch.tokens.shape[0], BATCH_SIZE):
        toks = batch.tokens[i:i + BATCH_SIZE]
        tgts = batch.targets[i:i + BATCH_SIZE]
        n = toks.shape[0]

        out = model_fn(toks)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise ValueError("model_fn must return a (logits, attn) tuple")
        logits, attn = out

        logits = np.asarray(logits, dtype=np.float64)
        if logits.shape != (n, seq_len, VOCAB_SIZE):
            raise ValueError(
                f"model_fn logits has shape {logits.shape}, "
                f"expected ({n}, {seq_len}, {VOCAB_SIZE})"
            )
        preds = np.argmax(logits, axis=-1)  # (n, seq_len)
        correct += int(np.sum(preds == tgts))
        total += int(preds.size)

        attn = _normalise_attn(attn, n, seq_len)
        # Mass each query position i places on its mirror key position.
        mirror_mass = attn[:, np.arange(seq_len), mirror]  # (n, seq_len)
        attn_mass_sum += float(mirror_mass.sum())
        attn_mass_count += int(mirror_mass.size)

    accuracy = correct / max(total, 1)
    mirror_attn_mass = attn_mass_sum / max(attn_mass_count, 1)

    # Data-driven no-mechanism reference: the identity (no-reversal) predictor,
    # which guesses the token at the SAME position. Computed from the data only,
    # so it is fully model-agnostic. Equals ~1/vocab in expectation off-centre.
    identity_acc = float(np.mean(batch.targets == batch.tokens))

    return {
        "seq_len": int(seq_len),
        "accuracy": float(accuracy),
        "mirror_attn_mass": float(mirror_attn_mass),
        "identity_baseline_accuracy": identity_acc,
        "num_sequences": int(batch.tokens.shape[0]),
    }


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run model_fn over the canonical condition and the length-generalisation
    sweep. Returns the payload dict matching benchmark.score's expected shape.
    """
    sweep = []
    for k, seq_len in enumerate(SEQ_LEN_SWEEP):
        # Per-slice seed keeps each length reproducible and independent.
        batch = _make_slice(seq_len, NUM_SEQUENCES, EVAL_SEED + 1000 * (k + 1))
        sweep.append(_eval_slice(model_fn, batch))

    return {
        "version": 1,
        "model_name": "synthetic_attention_reverse",
        "config": {
            "vocab_size": VOCAB_SIZE,
            "batch_size": BATCH_SIZE,
            "canonical_seq_len": CANONICAL_SEQ_LEN,
            "mask_type": "bidirectional",
            "num_sequences": NUM_SEQUENCES,
            "eval_seed": EVAL_SEED,
        },
        "vocab_size": VOCAB_SIZE,
        "seq_len_sweep": list(SEQ_LEN_SWEEP),
        "canonical_idx": CANONICAL_IDX,
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """
    Return a model_fn (the real signature) whose body emits random logits and a
    random row-normalised attention pattern. Pure NumPy; used by the smoke test.
    """
    rng = np.random.default_rng(0)

    def _random_fn(tokens: np.ndarray):
        tokens = np.asarray(tokens)
        n, seq_len = tokens.shape
        logits = rng.normal(size=(n, seq_len, VOCAB_SIZE)).astype(np.float32)
        raw = rng.random((n, seq_len, seq_len)).astype(np.float32)
        attn = raw / raw.sum(axis=-1, keepdims=True)
        return logits, attn

    return _random_fn
