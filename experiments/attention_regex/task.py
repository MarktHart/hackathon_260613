"""Task: synthetic attention-as-regex matching.

A model_fn is handed an (wildcard-capable) token pattern, a token embedding
matrix, and an embedded token sequence (the residual stream). It must emit
per-position logits that concentrate attention on the positions where the
pattern *finishes matching* the sequence. We measure how sharply attention
separates true match-end positions from the rest, swept over pattern length.

Everything here is pure NumPy — no torch, no GPU, no I/O. The pipeline smoke
test runs `evaluate(random_model_fn())` then `benchmark.score(...)`.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

# ModelFn signature, documented in README.md:
#   model_fn(pattern, embed, residual) -> logits
#     pattern  : (L,) int array; token ids in [0, V), with -1 marking a wildcard
#     embed    : (V, d) float array; row v is the embedding of token id v
#     residual : (N, d) float array; the embedded token sequence + noise
#   returns    : (N,) float array of per-position logits (softmaxed downstream)
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# --- Canonical measurement condition (see README.md) ---
D = 64                 # embedding / residual dimension
VOCAB_SIZE = 8         # alphabet size
N_POSITIONS = 120      # sequence length
N_PLANTS = 6           # planted matches injected per sequence
WILDCARD_PROB = 0.25   # per-pattern-position probability of a wildcard
SIGNAL = 2.0           # token-embedding amplitude in the residual stream
NOISE = 0.5            # residual-stream noise scale
CANONICAL_LENGTH = 3
LENGTH_SWEEP = [1, 2, 3, 4, 5, 6]
N_SEEDS = 10
EVAL_SEED = 42


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: many (pattern, embed, residual, labels) tuples.

    Each list has len(LENGTH_SWEEP) * N_SEEDS entries, aligned by index.
    """
    patterns: list      # (L,) int arrays, -1 == wildcard
    embeds: list        # (V, d) float arrays
    residuals: list     # (N, d) float arrays
    labels: list        # (N,) bool arrays — True where a match ends
    lengths: list       # pattern length (int) for each entry


def _match_end_labels(seq: np.ndarray, pattern: np.ndarray) -> np.ndarray:
    """Boolean (N,) array: True at position i if the pattern ends a match at i."""
    n = seq.shape[0]
    L = pattern.shape[0]
    labels = np.zeros(n, dtype=bool)
    for i in range(L - 1, n):
        window = seq[i - L + 1: i + 1]
        ok = True
        for j in range(L):
            p = int(pattern[j])
            if p != -1 and int(window[j]) != p:
                ok = False
                break
        labels[i] = ok
    return labels


def generate(seed: int = 0) -> Batch:
    """Deterministic batch for the canonical length sweep.

    Same `seed` -> identical batch. Produces len(LENGTH_SWEEP) * N_SEEDS
    evaluation conditions.
    """
    d, V, n = D, VOCAB_SIZE, N_POSITIONS

    patterns: list = []
    embeds: list = []
    residuals: list = []
    labels: list = []
    lengths: list = []

    for li, L in enumerate(LENGTH_SWEEP):
        for s in range(N_SEEDS):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(li) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))

            # Token embedding matrix, unit-norm rows.
            E = r.normal(size=(V, d))
            E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-8)

            # Build the pattern: token ids in [0, V) with some wildcards (-1).
            pattern = r.integers(0, V, size=L).astype(np.int64)
            wildcard = r.random(L) < WILDCARD_PROB
            # Keep at least one concrete token so the pattern is meaningful.
            if wildcard.all() and L > 0:
                wildcard[int(r.integers(0, L))] = False
            pattern[wildcard] = -1

            # Random base sequence.
            seq = r.integers(0, V, size=n).astype(np.int64)

            # Plant guaranteed matches at distinct start positions.
            if L <= n:
                max_start = n - L
                n_plant = min(N_PLANTS, max_start + 1)
                starts = r.choice(max_start + 1, size=n_plant, replace=False)
                for st in starts:
                    st = int(st)
                    for j in range(L):
                        p = int(pattern[j])
                        # Wildcards get a random concrete token at plant time.
                        seq[st + j] = p if p != -1 else int(r.integers(0, V))

            label = _match_end_labels(seq, pattern)

            # Residual stream: embedded tokens + noise.
            residual = E[seq] * SIGNAL + r.normal(size=(n, d)) * NOISE

            patterns.append(pattern.astype(np.int64))
            embeds.append(E.astype(np.float32))
            residuals.append(residual.astype(np.float32))
            labels.append(label.astype(bool))
            lengths.append(int(L))

    return Batch(
        patterns=patterns,
        embeds=embeds,
        residuals=residuals,
        labels=labels,
        lengths=lengths,
    )


def _sharpness(scores: np.ndarray, label: np.ndarray) -> float:
    """Normalised separation of scores on match positions vs the rest, in [0, 1]."""
    if np.any(label) and np.any(~label):
        mean_on = float(scores[label].mean())
        mean_off = float(scores[~label].mean())
        denom = max(abs(mean_on), 1e-8)
        return max(0.0, min(1.0, (mean_on - mean_off) / denom))
    return 0.0


def _last_concrete_token(pattern: np.ndarray) -> int:
    """Rightmost non-wildcard token id, or -1 if the pattern is all wildcards."""
    for j in range(pattern.shape[0] - 1, -1, -1):
        if int(pattern[j]) != -1:
            return int(pattern[j])
    return -1


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload."""
    batch = generate(seed=EVAL_SEED)

    by_len: dict = {L: [] for L in LENGTH_SWEEP}
    base_by_len: dict = {L: [] for L in LENGTH_SWEEP}

    for pattern, embed, residual, label, L in zip(
        batch.patterns, batch.embeds, batch.residuals, batch.labels, batch.lengths
    ):
        n_positions = residual.shape[0]

        # --- Attempt's model ---
        logits = np.asarray(
            model_fn(pattern, embed, residual), dtype=np.float64
        ).reshape(-1)
        if logits.shape != (n_positions,):
            raise ValueError(
                f"model_fn returned shape {logits.shape}, expected ({n_positions},)"
            )
        attn = np.exp(logits - logits.max())
        attn = attn / attn.sum()

        threshold = 1.0 / n_positions  # uniform-attention level
        pred = attn > threshold

        tp = int(np.sum(pred & label))
        fp = int(np.sum(pred & ~label))
        fn = int(np.sum(~pred & label))
        tn = int(np.sum(~pred & ~label))

        by_len[L].append({
            "match_sharpness": _sharpness(attn, label),
            "false_positive_rate": fp / max(fp + tn, 1),
            "false_negative_rate": fn / max(fn + tp, 1),
        })

        # --- Linear baseline: attend by similarity to the pattern's last
        # concrete token only (no sequential composition / no preceding
        # context). Strawman that should degrade as pattern length grows.
        t_last = _last_concrete_token(pattern)
        if t_last >= 0:
            base_score = residual @ embed[t_last]
        else:
            base_score = np.zeros(n_positions, dtype=np.float64)
        base_by_len[L].append(_sharpness(base_score, label))

    sweep = []
    linear_baseline = []
    for L in LENGTH_SWEEP:
        recs = by_len[L]
        sweep.append({
            "length": int(L),
            "match_sharpness": float(np.mean([r["match_sharpness"] for r in recs])),
            "false_positive_rate": float(np.mean([r["false_positive_rate"] for r in recs])),
            "false_negative_rate": float(np.mean([r["false_negative_rate"] for r in recs])),
            "n_seeds": len(recs),
        })
        bvals = base_by_len[L]
        linear_baseline.append({
            "length": int(L),
            "match_sharpness": float(np.mean(bvals)) if bvals else 0.0,
            "n_seeds": len(bvals),
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_regex",
        "d": D,
        "vocab_size": VOCAB_SIZE,
        "n_positions": N_POSITIONS,
        "canonical_length": CANONICAL_LENGTH,
        "length_sweep": list(LENGTH_SWEEP),
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


def random_model_fn() -> ModelFn:
    """A model_fn with the real signature that emits random logits.

    Pure NumPy; used by the pipeline smoke test.
    """
    rng = np.random.default_rng(0)

    def _random_fn(pattern: np.ndarray, embed: np.ndarray,
                   residual: np.ndarray) -> np.ndarray:
        n_positions = np.asarray(residual).shape[0]
        return rng.normal(size=n_positions).astype(np.float32)

    return _random_fn
