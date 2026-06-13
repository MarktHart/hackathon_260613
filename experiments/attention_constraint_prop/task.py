"""
Synthetic bracket-matching constraint-propagation task.

Question: do attention heads propagate syntactic constraints (matched
open/close brackets) across positions, and how does the fidelity vary with the
positional distance between the constrained tokens?

Exports
-------
Batch          : frozen dataclass holding generated sequences + ground truth.
generate(seed) : deterministic batch generator.
evaluate(fn)   : runs a model_fn over the batch, returns the benchmark payload.
random_model_fn(): a NumPy model_fn that returns uniform attention (the
                   no-mechanism baseline used by the smoke test).

model_fn contract
-----------------
    model_fn(input_ids: np.ndarray[int32, (batch, seq_len)])
        -> np.ndarray[float32, (batch, n_layers, n_heads, seq_len, seq_len)]

The returned array holds post-softmax attention weights; `A[b, l, h, i, j]` is
the weight query position `i` places on key position `j`. n_layers and n_heads
are model-dependent and inferred from the shape.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# ---- Vocabulary -----------------------------------------------------------
N_FILLER = 100                       # filler tokens 0..99
OPEN_A, CLOSE_A = 100, 101
OPEN_B, CLOSE_B = 102, 103
VOCAB_SIZE = 104

BRACKETS = {
    0: (OPEN_A, CLOSE_A),            # type 0 == "A"
    1: (OPEN_B, CLOSE_B),            # type 1 == "B"
}

# Distances at which constraint pairs are deliberately placed.  Guarantees the
# canonical distance (4) appears in the sweep regardless of the seed.
DISTANCES = (1, 2, 4, 8, 12, 16)

# ---- Canonical configuration ---------------------------------------------
SEQ_LEN = 32
NUM_SEQUENCES = 500
CONSTRAINT_TYPES = 2
CANONICAL_DISTANCE = 4
SEED = 0

ModelOutput = np.ndarray
ModelFn = Callable[[np.ndarray], ModelOutput]

# A directed constrained entry: (query_pos, key_pos, distance).
DirectedEntry = Tuple[int, int, int]


@dataclass(frozen=True)
class Batch:
    """A generated batch plus its ground-truth constraint graph."""
    input_ids: np.ndarray                      # int32 [batch, seq_len]
    # Per-sequence directed constrained entries: for a matched pair (o, c)
    # we record both (o, c, d) and (c, o, d) where d = |o - c|.
    constraints: List[List[DirectedEntry]]
    seq_len: int = SEQ_LEN
    num_sequences: int = NUM_SEQUENCES
    constraint_types: int = CONSTRAINT_TYPES
    canonical_distance: int = CANONICAL_DISTANCE
    seed: int = SEED
    vocab_size: int = VOCAB_SIZE


# ---- Generator ------------------------------------------------------------
def _generate_one(rng: np.random.Generator, seq_len: int) -> Tuple[np.ndarray, List[DirectedEntry]]:
    occupied = np.zeros(seq_len, dtype=bool)
    tokens = rng.integers(0, N_FILLER, size=seq_len).astype(np.int32)
    directed: List[DirectedEntry] = []

    for ctype in range(CONSTRAINT_TYPES):
        open_tok, close_tok = BRACKETS[ctype]
        n_pairs = int(rng.integers(2, 5))            # 2..4 pairs of this type
        for _ in range(n_pairs):
            placed = False
            for _try in range(40):
                d = int(rng.choice(DISTANCES))
                if d >= seq_len:
                    continue
                o = int(rng.integers(0, seq_len - d))
                c = o + d
                if occupied[o] or occupied[c]:
                    continue
                occupied[o] = occupied[c] = True
                tokens[o] = open_tok
                tokens[c] = close_tok
                directed.append((o, c, d))
                directed.append((c, o, d))
                placed = True
                break
            # if not placed after many tries, silently skip this pair
            if not placed:
                continue
    return tokens, directed


def generate(seed: int = SEED, num_sequences: int = NUM_SEQUENCES,
             seq_len: int = SEQ_LEN) -> Batch:
    """Deterministic for a given seed: same seed -> identical batch."""
    rng = np.random.default_rng(seed)
    all_tokens = []
    all_constraints: List[List[DirectedEntry]] = []
    for _ in range(num_sequences):
        toks, directed = _generate_one(rng, seq_len)
        all_tokens.append(toks)
        all_constraints.append(directed)
    return Batch(
        input_ids=np.stack(all_tokens, axis=0).astype(np.int32),
        constraints=all_constraints,
        seq_len=seq_len,
        num_sequences=num_sequences,
        seed=seed,
    )


# ---- Evaluator ------------------------------------------------------------
def _flatten_constraints(batch: Batch) -> Dict[str, np.ndarray]:
    b_idx, i_idx, j_idx, d_idx = [], [], [], []
    for b, entries in enumerate(batch.constraints):
        for (i, j, d) in entries:
            b_idx.append(b)
            i_idx.append(i)
            j_idx.append(j)
            d_idx.append(d)
    return {
        "b": np.asarray(b_idx, dtype=np.int64),
        "i": np.asarray(i_idx, dtype=np.int64),
        "j": np.asarray(j_idx, dtype=np.int64),
        "d": np.asarray(d_idx, dtype=np.int64),
    }


def evaluate(model_fn: ModelFn, batch: Optional[Batch] = None) -> dict:
    """Run ``model_fn`` over the batch and build the benchmark payload."""
    if batch is None:
        batch = generate(seed=SEED)

    attn = np.asarray(model_fn(batch.input_ids), dtype=np.float32)
    if attn.ndim != 5:
        raise ValueError(
            f"model_fn output must be 5D [batch, n_layers, n_heads, seq, seq], "
            f"got shape {attn.shape}"
        )
    B, L, H, Sq, Sk = attn.shape
    if B != batch.input_ids.shape[0]:
        raise ValueError(
            f"attention batch dim {B} != input batch {batch.input_ids.shape[0]}"
        )
    if Sq != batch.seq_len or Sk != batch.seq_len:
        raise ValueError(
            f"attention seq dims ({Sq}, {Sk}) != seq_len {batch.seq_len}"
        )

    flat = _flatten_constraints(batch)
    distances = sorted(int(d) for d in np.unique(flat["d"]))

    sweep = []
    for d in distances:
        mask = flat["d"] == d
        bb, ii, jj = flat["b"][mask], flat["i"][mask], flat["j"][mask]

        heads = []
        head_aligns = np.zeros((L, H), dtype=np.float64)
        for l in range(L):
            for h in range(H):
                if bb.size:
                    vals = attn[bb, l, h, ii, jj]
                    a = float(np.mean(vals))
                else:
                    a = 0.0
                head_aligns[l, h] = a
                heads.append({"layer": int(l), "head": int(h), "alignment": a})

        flat_idx = int(np.argmax(head_aligns))
        best_l, best_h = divmod(flat_idx, H)
        sweep.append({
            "distance": int(d),
            "n_entries": int(bb.size),
            "heads": heads,
            "mean_alignment": float(np.mean(head_aligns)),
            "max_alignment": float(np.max(head_aligns)),
            "best_head": {
                "layer": int(best_l),
                "head": int(best_h),
                "alignment": float(head_aligns[best_l, best_h]),
            },
        })

    payload = {
        "version": 1,
        "config": {
            "seq_len": batch.seq_len,
            "num_sequences": batch.num_sequences,
            "constraint_types": batch.constraint_types,
            "canonical_distance": batch.canonical_distance,
            "seed": batch.seed,
        },
        "model_info": {"n_layers": int(L), "n_heads": int(H)},
        "sweep": sweep,
    }
    return payload


# ---- Random / baseline model ---------------------------------------------
def random_model_fn() -> ModelFn:
    """A NumPy model_fn returning uniform (1/seq_len) attention everywhere.

    Pure NumPy, no torch, no GPU.  Used by the smoke test and as the
    no-mechanism baseline; its alignment equals the random baseline 1/seq_len.
    """
    def _fn(input_ids: np.ndarray) -> np.ndarray:
        input_ids = np.asarray(input_ids)
        batch, seq = input_ids.shape
        n_layers, n_heads = 2, 3
        attn = np.full(
            (batch, n_layers, n_heads, seq, seq),
            1.0 / float(seq),
            dtype=np.float32,
        )
        return attn
    return _fn
