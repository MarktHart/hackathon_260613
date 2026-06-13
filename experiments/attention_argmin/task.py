"""Task for the attention_argmin goal.

Synthetic, pure-NumPy generator + evaluator. An attempt supplies a `model_fn`
that, given the keys, values and a query for a single sequence, returns an
attention distribution over positions. A perfect argmin head puts all of its
mass on the position of the minimum value.

Exports:
    generate(seed) -> Batch          deterministic data
    evaluate(model_fn) -> dict        payload consumed by benchmark.score
    random_model_fn() -> ModelFn      shape-correct dummy (uniform attention)

No torch, no GPU, no I/O.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np

# model_fn(keys (seq_len, key_dim), values (seq_len,), query (key_dim,))
#   -> attention weights (seq_len,)
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# ----- canonical configuration (fixed for every attempt) -----
SEQ_LEN = 64
KEY_DIM = 32
N_SEQ_PER_GAP = 200
GAPS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
CANONICAL_GAP = 0.5
SEED = 0


@dataclass(frozen=True)
class Batch:
    keys: np.ndarray              # (n_total, seq_len, key_dim)
    values: np.ndarray            # (n_total, seq_len)
    query: np.ndarray             # (key_dim,)
    argmin_positions: np.ndarray  # (n_total,) — true argmin index per sequence
    gap_index: np.ndarray         # (n_total,) — which entry of GAPS each row uses


def generate(seed: int = SEED) -> Batch:
    """Deterministic: same seed -> identical Batch.

    Builds `N_SEQ_PER_GAP` sequences for each gap in `GAPS`. Within a sequence
    the values are Uniform(-1, 1) distractors, with the minimum placed at a
    random position equal to `-1 - gap` and the runner-up at `-1 + gap`. The
    `gap` is the margin separating the true minimum from the next-smallest
    value: small gaps make the argmin harder to resolve.
    """
    rng = np.random.default_rng(seed)

    # Shared (unit-norm) keys and a fixed query across all sequences.
    keys = rng.normal(size=(SEQ_LEN, KEY_DIM)).astype(np.float32)
    keys = keys / np.linalg.norm(keys, axis=1, keepdims=True)
    query = np.zeros(KEY_DIM, dtype=np.float32)
    query[0] = 1.0

    all_keys, all_values, all_argmin, all_gap_idx = [], [], [], []
    for gi, gap in enumerate(GAPS):
        for _ in range(N_SEQ_PER_GAP):
            values = rng.uniform(-1.0, 1.0, size=SEQ_LEN).astype(np.float32)
            pos_min, pos_second = rng.choice(SEQ_LEN, size=2, replace=False)
            values[pos_min] = -1.0 - gap
            values[pos_second] = -1.0 + gap
            all_keys.append(keys)
            all_values.append(values)
            all_argmin.append(int(pos_min))
            all_gap_idx.append(gi)

    return Batch(
        keys=np.stack(all_keys),
        values=np.stack(all_values),
        query=query,
        argmin_positions=np.array(all_argmin, dtype=np.int32),
        gap_index=np.array(all_gap_idx, dtype=np.int32),
    )


def _run_model(model_fn: ModelFn, batch: Batch) -> np.ndarray:
    """Apply model_fn per sequence -> (n_total, seq_len) attention array."""
    n_total = batch.values.shape[0]
    attn = np.zeros((n_total, SEQ_LEN), dtype=np.float64)
    for i in range(n_total):
        w = np.asarray(
            model_fn(batch.keys[i], batch.values[i], batch.query),
            dtype=np.float64,
        ).reshape(-1)
        if w.shape[0] != SEQ_LEN:
            raise ValueError(
                f"model_fn returned shape {w.shape}, expected ({SEQ_LEN},)"
            )
        attn[i] = w
    return attn


def _record_for_gap(attn: np.ndarray, argmin: np.ndarray, gap: float) -> dict:
    """Reduce one gap-slice of attention to a scalar record."""
    n = attn.shape[0]
    rows = np.arange(n)
    at_min = attn[rows, argmin]                       # (n,)
    total = attn.sum(axis=1)                          # (n,)
    others_sum = total - at_min
    # Mean attention *per non-argmin position* (so uniform attention -> equal,
    # giving a sharpness ratio of exactly 1.0 for the strawman).
    others_per_pos = others_sum / max(SEQ_LEN - 1, 1)

    attn_argmax = np.argmax(attn, axis=1)
    argmax_correct = float(np.mean(attn_argmax == argmin))

    return {
        "gap": float(gap),
        "sequences": int(n),
        "seq_len": int(attn.shape[1]),
        "attn_at_min": float(np.mean(at_min)),
        "attn_at_others": float(np.mean(others_per_pos)),
        "argmax_correct": argmax_correct,
    }


def _build_side(attn: np.ndarray, batch: Batch) -> dict:
    """Build {canonical, sweep} from a full (n_total, seq_len) attention array."""
    sweep, canonical = [], None
    for gi, gap in enumerate(GAPS):
        mask = batch.gap_index == gi
        rec = _record_for_gap(attn[mask], batch.argmin_positions[mask], gap)
        sweep.append(rec)
        if gap == CANONICAL_GAP:
            canonical = rec
    if canonical is None:  # defensive; CANONICAL_GAP is always in GAPS
        canonical = sweep[0]
    return {"canonical": canonical, "sweep": sweep}


def _uniform_attention(keys, values, query):
    """No-mechanism reference: spread attention evenly over all positions."""
    seq_len = keys.shape[0]
    return np.full(seq_len, 1.0 / seq_len, dtype=np.float64)


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch and return the scoring payload.

    The uniform-attention strawman is measured on the *same* batch to give the
    linear baseline the benchmark contrasts against.
    """
    batch = generate(SEED)

    attn = _run_model(model_fn, batch)
    base_attn = _run_model(_uniform_attention, batch)

    side = _build_side(attn, batch)
    base_side = _build_side(base_attn, batch)

    return {
        "version": 1,
        "canonical": side["canonical"],
        "sweep": side["sweep"],
        "linear_baseline": {
            "canonical": base_side["canonical"],
            "sweep": base_side["sweep"],
        },
        "model_config": {
            "seq_len": SEQ_LEN,
            "key_dim": KEY_DIM,
            "n_seq_per_gap": N_SEQ_PER_GAP,
            "gaps": list(GAPS),
            "canonical_gap": CANONICAL_GAP,
            "seed": SEED,
        },
    }


def random_model_fn() -> ModelFn:
    """A shape-correct dummy model_fn (uniform attention). Pure NumPy."""
    def _fn(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
        seq_len = keys.shape[0]
        return np.full(seq_len, 1.0 / seq_len, dtype=np.float64)
    return _fn
