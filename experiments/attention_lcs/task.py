"""Synthetic LCS-alignment task for attention mechanistic interpretability.

Exports:
    generate(seed) -> Batch        deterministic data
    evaluate(model_fn) -> dict     run a model, return the benchmark payload
    random_model_fn() -> ModelFn   a shape-correct no-op model (uniform attention)

Pure NumPy. No torch, no GPU, no I/O.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Canonical measurement condition (see README.md).
SEQ_LEN = 16
VOCAB_SIZE = 8
NUM_EXAMPLES = 256
EVAL_SEED = 0

# model_fn(seq_a, seq_b) -> attn[batch, n_heads, seq_len, seq_len]
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    seq_a: np.ndarray                     # [num_examples, seq_len] int32 (queries)
    seq_b: np.ndarray                     # [num_examples, seq_len] int32 (keys)
    # match_keys[b][q] = sorted list of key positions in B that are the LCS
    # partners of query position q in A (empty if q is not in the LCS).
    match_keys: list[list[list[int]]]
    config: dict


def _lcs_alignment(seq_a: np.ndarray, seq_b: np.ndarray) -> list[list[int]]:
    """
    One canonical LCS alignment between seq_a (queries) and seq_b (keys).

    Returns match_keys: for each position i in seq_a, the list of positions j
    in seq_b matched to it in the traced-back LCS (0 or 1 entries here, since a
    single trace yields a one-to-one matching).
    """
    n, m = len(seq_a), len(seq_b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        ai = seq_a[i - 1]
        row_prev = dp[i - 1]
        row = dp[i]
        for j in range(1, m + 1):
            if ai == seq_b[j - 1]:
                row[j] = row_prev[j - 1] + 1
            elif row_prev[j] >= row[j - 1]:
                row[j] = row_prev[j]
            else:
                row[j] = row[j - 1]

    match_keys: list[list[int]] = [[] for _ in range(n)]
    i, j = n, m
    while i > 0 and j > 0:
        if seq_a[i - 1] == seq_b[j - 1]:
            match_keys[i - 1].append(j - 1)
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    for ml in match_keys:
        ml.sort()
    return match_keys


def generate(seed: int = 0) -> Batch:
    """Deterministic batch. Same seed -> same data."""
    rng = np.random.default_rng(seed)
    seq_a = rng.integers(0, VOCAB_SIZE, size=(NUM_EXAMPLES, SEQ_LEN), dtype=np.int32)
    seq_b = rng.integers(0, VOCAB_SIZE, size=(NUM_EXAMPLES, SEQ_LEN), dtype=np.int32)

    match_keys: list[list[list[int]]] = []
    for b in range(NUM_EXAMPLES):
        match_keys.append(_lcs_alignment(seq_a[b], seq_b[b]))

    config = {
        "seq_len": SEQ_LEN,
        "vocab_size": VOCAB_SIZE,
        "num_examples": NUM_EXAMPLES,
        "seed": seed,
    }
    return Batch(seq_a=seq_a, seq_b=seq_b, match_keys=match_keys, config=config)


def _validate_attn(attn: np.ndarray, batch: Batch) -> None:
    if not isinstance(attn, np.ndarray):
        raise ValueError(f"model_fn must return a numpy array, got {type(attn).__name__}")
    if attn.ndim != 4:
        raise ValueError(
            f"model_fn must return 4D [batch, n_heads, seq_len, seq_len], got shape {attn.shape}"
        )
    bsz, n_heads, q_len, k_len = attn.shape
    if bsz != batch.seq_a.shape[0]:
        raise ValueError(f"batch mismatch: model {bsz}, expected {batch.seq_a.shape[0]}")
    if n_heads < 1:
        raise ValueError(f"n_heads must be >= 1, got {n_heads}")
    if q_len != batch.seq_a.shape[1] or k_len != batch.seq_b.shape[1]:
        raise ValueError(
            f"seq_len mismatch: model q={q_len} k={k_len}, "
            f"expected q={batch.seq_a.shape[1]} k={batch.seq_b.shape[1]}"
        )
    if not np.all(np.isfinite(attn)):
        raise ValueError("model_fn returned non-finite attention weights")
    if np.any(attn < -1e-4):
        raise ValueError("model_fn returned negative attention weights")
    sums = attn.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-3):
        bad = float(np.max(np.abs(sums - 1.0)))
        raise ValueError(f"attention must sum to 1 over keys; max deviation {bad:.4g}")


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn on the canonical batch and return the benchmark payload."""
    batch = generate(seed=EVAL_SEED)
    attn = np.asarray(model_fn(batch.seq_a, batch.seq_b), dtype=np.float64)
    _validate_attn(attn, batch)

    bsz, n_heads, _q_len, k_len = attn.shape

    # Precompute, per example, the flat list of (query_pos, [match_keys]).
    # random_baseline_mass depends only on data, so compute it once.
    scored: list[tuple[int, int, np.ndarray]] = []  # (example, q_pos, key_idx array)
    baseline_total = 0.0
    n_scored = 0
    for b in range(bsz):
        mk = batch.match_keys[b]
        for q_pos, keys in enumerate(mk):
            if not keys:
                continue
            key_arr = np.asarray(keys, dtype=np.int64)
            scored.append((b, q_pos, key_arr))
            baseline_total += key_arr.shape[0] / k_len
            n_scored += 1

    random_baseline_mass = (baseline_total / n_scored) if n_scored > 0 else 0.0

    sweep = []
    for h in range(n_heads):
        mass_total = 0.0
        for (b, q_pos, key_arr) in scored:
            mass_total += float(attn[b, h, q_pos, key_arr].sum())
        lcs_mass = (mass_total / n_scored) if n_scored > 0 else 0.0
        sweep.append({
            "head": h,
            "lcs_attention_mass": float(lcs_mass),
            "lcs_lift": float(lcs_mass - random_baseline_mass),
            "n_query_positions": int(n_scored),
        })

    config = dict(batch.config)
    config["n_heads"] = int(n_heads)

    return {
        "version": 1,
        "config": config,
        "random_baseline_mass": float(random_baseline_mass),
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A shape-correct no-op model: uniform attention over keys, 4 heads.

    Same signature as a real model_fn. Pure NumPy. Used by the smoke test.
    """
    def _fn(seq_a: np.ndarray, seq_b: np.ndarray) -> np.ndarray:
        bsz = seq_a.shape[0]
        k_len = seq_b.shape[1]
        q_len = seq_a.shape[1]
        n_heads = 4
        attn = np.full(
            (bsz, n_heads, q_len, k_len), 1.0 / k_len, dtype=np.float32
        )
        return attn
    return _fn


if __name__ == "__main__":
    payload = evaluate(random_model_fn())
    print("version:", payload["version"])
    print("config:", payload["config"])
    print("random_baseline_mass:", round(payload["random_baseline_mass"], 4))
    print("n heads:", len(payload["sweep"]))
    for rec in payload["sweep"]:
        print("  head", rec["head"],
              "mass", round(rec["lcs_attention_mass"], 4),
              "lift", round(rec["lcs_lift"], 4))
