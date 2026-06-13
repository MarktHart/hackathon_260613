import numpy as np
from dataclasses import dataclass
from typing import Callable

# ----------------------------------------------------------------------------
# model_fn contract (see README.md)
#
# A model_fn takes a batch of sequences for ONE sequence length and returns the
# attention logits a "sorting head" assigns from each output (query) slot to
# each input (key) position:
#
#     model_fn(values: np.ndarray[B, L] float32) -> np.ndarray[B, L, L] float
#
# Row i of the returned [L, L] matrix is the (unnormalised) logits over input
# positions for output slot i, where output slot i should attend to the input
# position that holds the i-th smallest value. `evaluate` softmaxes each row, so
# attempts may return raw logits OR a row-stochastic matrix (softmax of an
# already-normalised distribution is monotone and preserves the argmax / mass
# structure we score). Pure NumPy in, pure NumPy out — no torch, no GPU.
# ----------------------------------------------------------------------------
ModelFn = Callable[[np.ndarray], np.ndarray]

# Canonical measurement condition.
SWEEP_LENGTHS = (4, 8, 16, 32)
CANONICAL_LENGTH = 8
N_SEQUENCES = 256
EVAL_SEED = 42


@dataclass(frozen=True)
class Batch:
    """Evaluation data: one array of distinct-valued sequences per length."""
    lengths: tuple                       # the sweep lengths, ascending
    sequences: dict                      # length -> np.ndarray[N_SEQUENCES, L] float32
    n_sequences: int


def generate(seed: int = 0) -> Batch:
    """
    Deterministic batch for the canonical sweep. Same seed -> same data.

    For each length L in SWEEP_LENGTHS we draw N_SEQUENCES sequences of L
    continuous values (distinct with probability 1). The "sorting" target for a
    sequence is argsort(values): output slot i should attend to the input
    position holding the i-th smallest value.
    """
    sequences: dict = {}
    for li, L in enumerate(SWEEP_LENGTHS):
        # Distinct, well-separated per-length RNG stream.
        seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                  + np.uint64(li) * np.uint64(9_973)
                  + np.uint64(L)) & np.uint64(0xFFFFFFFF)
        r = np.random.default_rng(int(seed_i))
        vals = r.random(size=(N_SEQUENCES, int(L))).astype(np.float32)
        sequences[int(L)] = vals
    return Batch(
        lengths=tuple(int(L) for L in SWEEP_LENGTHS),
        sequences=sequences,
        n_sequences=N_SEQUENCES,
    )


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis, numerically stable."""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=-1, keepdims=True), 1e-12, None)


def _sortedness(out: np.ndarray) -> float:
    """Fraction of adjacent output pairs that are in non-decreasing order."""
    if out.shape[1] < 2:
        return 1.0
    ordered = out[:, :-1] <= out[:, 1:] + 1e-9
    return float(ordered.mean())


def evaluate(model_fn: ModelFn) -> dict:
    """
    Run `model_fn` over the canonical batch and return a payload dict matching
    the benchmark.score contract exactly.
    """
    batch = generate(seed=EVAL_SEED)

    sweep = []
    for L in batch.lengths:
        values = batch.sequences[L]            # [N, L]
        N = values.shape[0]

        # target_key[n, i] = input position holding the i-th smallest value.
        target_key = np.argsort(values, axis=1)  # [N, L]

        logits = np.asarray(model_fn(values), dtype=np.float64)
        if logits.shape != (N, L, L):
            raise ValueError(
                f"model_fn returned shape {logits.shape}, expected ({N}, {L}, {L})"
            )
        attn = np.stack([_softmax_rows(logits[n]) for n in range(N)])  # [N, L, L]

        # --- Attempt's sorting head ---
        # argmax over keys for each output slot.
        argmax_key = np.argmax(attn, axis=2)                # [N, L]
        sort_accuracy = float(np.mean(argmax_key == target_key))

        # mass placed on the correct target key.
        rows = np.arange(L)[None, :]
        target_mass = float(np.mean(
            attn[np.arange(N)[:, None], rows, target_key]
        ))

        # sortedness of the attention-mixed output sequence.
        out = np.einsum("nij,nj->ni", attn, values)         # [N, L]
        output_sortedness = _sortedness(out)

        # --- No-mechanism baselines under identical conditions ---
        # Sortedness of the raw (unsorted) input — "did nothing" reference,
        # ~0.5 for random data. (Uniform attention would collapse the output to
        # a constant and score a trivial 1.0, so it is NOT a useful reference.)
        unsorted_sortedness = _sortedness(values)
        # Expected argmax-hit rate of uniform/random attention.
        uniform_accuracy = 1.0 / float(L)

        sweep.append({
            "length": int(L),
            "sort_accuracy": sort_accuracy,
            "target_mass": target_mass,
            "output_sortedness": output_sortedness,
            "unsorted_sortedness": unsorted_sortedness,
            "uniform_accuracy": uniform_accuracy,
            "n_sequences": int(N),
        })

    return {
        "version": 1,
        "task": "attention_sort",
        "canonical_length": CANONICAL_LENGTH,
        "sweep_lengths": list(batch.lengths),
        "n_sequences": batch.n_sequences,
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """
    Return a model_fn with the real signature whose body emits random logits.
    Pure NumPy; used by the pipeline smoke test. Takes NO arguments and returns
    a callable matching the ModelFn contract.
    """
    rng = np.random.default_rng(0)

    def _random_fn(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values)
        N, L = values.shape
        return rng.normal(size=(N, L, L)).astype(np.float32)

    return _random_fn
