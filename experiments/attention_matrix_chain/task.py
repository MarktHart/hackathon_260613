"""Task definition for the `attention_matrix_chain` goal.

Synthetic, fully controlled, pure NumPy. Exports:

    generate(seed) -> Batch
    evaluate(model_fn) -> payload dict   (consumed verbatim by benchmark.score)
    random_model_fn() -> ModelFn         (reference model for the smoke test)

The goal asks whether a mechanism can reconstruct the *composed* two-hop
attention matrix  A_chain = A2 @ A1  from the two single-layer patterns, and
how that fidelity holds up as the attention rows become more peaked (where a
single-hop shortcut increasingly fails).
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable

# ModelFn signature (see README.md):
#   model_fn(A1, A2) -> predicted A_chain
#     A1, A2 : (num_heads, seq_len, seq_len) row-stochastic attention patterns
#     return : (num_heads, seq_len, seq_len) predicted composed attention
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# --- Canonical measurement condition (documented in README.md) ---
NUM_HEADS = 4
SEQ_LEN = 12
# Dirichlet concentration per attention row. Smaller -> more peaked/sparse
# rows -> composition matters more and the single-hop shortcut fails harder.
ALPHA_SWEEP = [0.1, 0.3, 1.0, 3.0, 10.0]
CANONICAL_ALPHA = 0.3
N_SEEDS = 8
EVAL_SEED = 42

_EPS = 1e-12


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: many (A1, A2, A_chain) triples across alpha x seed."""
    A1s: list          # list of (num_heads, seq, seq) row-stochastic
    A2s: list          # list of (num_heads, seq, seq) row-stochastic
    chains: list        # list of (num_heads, seq, seq) = A2 @ A1 (ground truth)
    alphas: list        # nominal Dirichlet alpha for each entry (float)
    config: dict        # canonical config, self-describing


def _row_stochastic(rng: np.random.Generator, alpha: float) -> np.ndarray:
    """Sample a (num_heads, seq, seq) row-stochastic attention pattern."""
    conc = np.full(SEQ_LEN, float(alpha))
    M = np.empty((NUM_HEADS, SEQ_LEN, SEQ_LEN), dtype=np.float64)
    for h in range(NUM_HEADS):
        for i in range(SEQ_LEN):
            M[h, i] = rng.dirichlet(conc)
    return M


def generate(seed: int = 0) -> Batch:
    """Deterministic batch for the canonical sweep.

    Produces len(ALPHA_SWEEP) * N_SEEDS conditions. `seed` shifts the per-entry
    RNG so the whole batch is reproducible for a given seed.
    """
    A1s: list = []
    A2s: list = []
    chains: list = []
    alphas: list = []

    for ai, alpha in enumerate(ALPHA_SWEEP):
        for s in range(N_SEEDS):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(ai) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))

            A1 = _row_stochastic(r, alpha)
            A2 = _row_stochastic(r, alpha)
            chain = np.matmul(A2, A1)  # (num_heads, seq, seq), row-stochastic

            A1s.append(A1.astype(np.float32))
            A2s.append(A2.astype(np.float32))
            chains.append(chain.astype(np.float32))
            alphas.append(float(alpha))

    config = {
        "num_heads": NUM_HEADS,
        "seq_len": SEQ_LEN,
        "alpha_sweep": list(ALPHA_SWEEP),
        "canonical_alpha": CANONICAL_ALPHA,
        "n_seeds": N_SEEDS,
        "seed": int(seed),
    }
    return Batch(A1s=A1s, A2s=A2s, chains=chains, alphas=alphas, config=config)


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    """Project an arbitrary (..., seq) array to row-stochastic distributions.

    Clip negatives, normalise each row to sum 1; all-zero rows become uniform.
    Keeps the metric well defined for any model output.
    """
    M = np.clip(np.asarray(M, dtype=np.float64), 0.0, None)
    row_sums = M.sum(axis=-1, keepdims=True)
    uniform = np.full_like(M, 1.0 / M.shape[-1])
    safe = np.where(row_sums < _EPS, 1.0, row_sums)
    out = np.where(row_sums < _EPS, uniform, M / safe)
    return out


def _fidelity(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean row fidelity = 1 - mean total-variation distance, in [0, 1]."""
    p = _normalize_rows(pred)
    q = _normalize_rows(true)
    tv = 0.5 * np.abs(p - q).sum(axis=-1)  # (num_heads, seq) in [0, 1]
    return float(1.0 - tv.mean())


def _row_kl(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean KL(true || pred) over rows, in nats (>= 0). Eps-smoothed."""
    p = _normalize_rows(pred) + _EPS
    q = _normalize_rows(true) + _EPS
    kl = (q * np.log(q / p)).sum(axis=-1)  # (num_heads, seq)
    return float(kl.mean())


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload."""
    batch = generate(seed=EVAL_SEED)

    expected_shape = (NUM_HEADS, SEQ_LEN, SEQ_LEN)
    by_alpha: dict = {a: [] for a in ALPHA_SWEEP}
    base_by_alpha: dict = {a: [] for a in ALPHA_SWEEP}

    for A1, A2, chain, alpha in zip(
        batch.A1s, batch.A2s, batch.chains, batch.alphas
    ):
        pred = np.asarray(model_fn(A1, A2), dtype=np.float64)
        if pred.shape != expected_shape:
            raise ValueError(
                f"model_fn returned shape {pred.shape}, expected {expected_shape}"
            )

        by_alpha[alpha].append({
            "fidelity": _fidelity(pred, chain),
            "kl": _row_kl(pred, chain),
        })
        # Single-hop baseline: ignore composition, predict layer-2 alone.
        base_by_alpha[alpha].append(_fidelity(A2, chain))

    sweep = []
    baseline = []
    for alpha in ALPHA_SWEEP:
        recs = by_alpha[alpha]
        sweep.append({
            "alpha": float(alpha),
            "chain_fidelity": float(np.mean([r["fidelity"] for r in recs])) if recs else 0.0,
            "row_kl": float(np.mean([r["kl"] for r in recs])) if recs else 0.0,
            "n_seeds": len(recs),
        })
        bvals = base_by_alpha[alpha]
        baseline.append({
            "alpha": float(alpha),
            "chain_fidelity": float(np.mean(bvals)) if bvals else 0.0,
            "n_seeds": len(bvals),
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_matrix_chain",
        "num_heads": NUM_HEADS,
        "seq_len": SEQ_LEN,
        "canonical_alpha": CANONICAL_ALPHA,
        "alpha_sweep": list(ALPHA_SWEEP),
        "sweep": sweep,
        "single_hop_baseline": baseline,
    }


def random_model_fn() -> ModelFn:
    """Return a model_fn (real signature) emitting random row-stochastic output.

    Pure NumPy; used by the pipeline smoke test.
    """
    rng = np.random.default_rng(0)

    def _random_fn(A1: np.ndarray, A2: np.ndarray) -> np.ndarray:
        shape = np.asarray(A1).shape
        logits = rng.normal(size=shape)
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)

    return _random_fn
