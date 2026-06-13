"""Task: attention_dtw.

Synthetic alignment task. Two sequences are time-warped views of the same
underlying random signal, with a known monotone ground-truth alignment. An
attempt's `model_fn` produces attention from each key position to the query
positions; we measure how well the best head's argmax path recovers the
ground-truth alignment, and how much that quality is retained as the warp
grows.

Exports:
    generate(seed) -> Batch
    evaluate(model_fn) -> payload dict   (shape consumed by benchmark.score)
    random_model_fn() -> ModelFn         (pure NumPy, for the smoke test)

Pure CPU / NumPy. No torch, no I/O, no network.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

# ----------------------------------------------------------------------
# Fixed configuration (bump benchmark.VERSION if any of these change)
# ----------------------------------------------------------------------
VERSION = 1
M = 16            # query sequence length
N = 20            # key sequence length
D = 8             # feature dimension
N_EXAMPLES = 24   # examples per warp slice
WARP_SWEEP = (0.0, 0.25, 0.5, 0.75)
CANONICAL_WARP = 0.5
NOISE = 0.1

ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Example:
    queries: np.ndarray  # (M, D)
    keys: np.ndarray     # (N, D)
    align: np.ndarray    # (N,) int, ground-truth query index per key position


@dataclass(frozen=True)
class Batch:
    warp_sweep: tuple
    canonical_warp: float
    m: int
    n: int
    d: int
    n_examples: int
    examples: Dict[float, List[Example]]


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
def _make_alignment(rng: np.random.Generator, warp: float) -> np.ndarray:
    """Monotone non-decreasing mapping key_index -> query_index in [0, M-1].

    At warp == 0 this is the straight diagonal. Larger warp bends the schedule.
    """
    incs = np.exp(warp * rng.standard_normal(N))
    cum = np.cumsum(incs)
    cum = cum - cum[0]
    if cum[-1] <= 0:
        frac = np.linspace(0.0, 1.0, N)
    else:
        frac = cum / cum[-1]
    align = np.round(frac * (M - 1)).astype(int)
    align = np.clip(align, 0, M - 1)
    align = np.maximum.accumulate(align)  # enforce monotone non-decreasing
    return align


def _make_example(rng: np.random.Generator, warp: float) -> Example:
    queries = rng.standard_normal((M, D))
    align = _make_alignment(rng, warp)
    keys = queries[align] + NOISE * rng.standard_normal((N, D))
    return Example(queries=queries, keys=keys, align=align)


def generate(seed: int = 0) -> Batch:
    """Deterministic for a given seed: same seed -> same batch."""
    examples: Dict[float, List[Example]] = {}
    for wi, warp in enumerate(WARP_SWEEP):
        rng = np.random.default_rng([int(seed), wi])
        examples[warp] = [_make_example(rng, warp) for _ in range(N_EXAMPLES)]
    return Batch(
        warp_sweep=WARP_SWEEP,
        canonical_warp=CANONICAL_WARP,
        m=M,
        n=N,
        d=D,
        n_examples=N_EXAMPLES,
        examples=examples,
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
def _coerce_attn(raw, num_heads_seen) -> np.ndarray:
    attn = np.asarray(raw, dtype=float)
    if attn.ndim == 2:
        attn = attn[None, :, :]
    if attn.ndim != 3:
        raise ValueError(
            f"model_fn must return a 2-D (N, M) or 3-D (heads, N, M) array; "
            f"got ndim={attn.ndim}"
        )
    H, n_, m_ = attn.shape
    if n_ != N or m_ != M:
        raise ValueError(
            f"model_fn output must have shape (heads, {N}, {M}); got {attn.shape}"
        )
    if num_heads_seen is not None and H != num_heads_seen:
        raise ValueError(
            f"model_fn changed head count between calls: {num_heads_seen} -> {H}"
        )
    if not np.all(np.isfinite(attn)):
        raise ValueError("model_fn output contains NaN or inf")
    return attn


def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over the synthetic batch and return the scoring payload."""
    batch = generate(0)

    diag_pred = np.round(np.arange(N) * (M - 1) / (N - 1)).astype(int)
    uniform_overlap = 1.0 / M

    num_heads = None
    sweep: List[dict] = []
    baseline: List[dict] = []

    for warp in batch.warp_sweep:
        exs = batch.examples[warp]
        best_overlaps: List[float] = []
        mean_overlaps: List[float] = []
        monotonicities: List[float] = []
        diag_overlaps: List[float] = []

        for ex in exs:
            attn = _coerce_attn(model_fn(ex.queries, ex.keys), num_heads)
            num_heads = attn.shape[0]

            preds = np.argmax(attn, axis=2)                        # (H, N)
            overlaps = np.mean(preds == ex.align[None, :], axis=1)  # (H,)
            best_h = int(np.argmax(overlaps))

            best_overlaps.append(float(overlaps[best_h]))
            mean_overlaps.append(float(np.mean(overlaps)))

            bp = preds[best_h]
            mono = float(np.mean(np.diff(bp) >= 0)) if N > 1 else 1.0
            monotonicities.append(mono)

            diag_overlaps.append(float(np.mean(diag_pred == ex.align)))

        sweep.append({
            "warp": float(warp),
            "best_head_overlap": float(np.mean(best_overlaps)),
            "mean_head_overlap": float(np.mean(mean_overlaps)),
            "monotonicity": float(np.mean(monotonicities)),
        })
        baseline.append({
            "warp": float(warp),
            "diagonal_overlap": float(np.mean(diag_overlaps)),
            "uniform_overlap": float(uniform_overlap),
        })

    return {
        "version": VERSION,
        "setup": "synthetic_dtw_alignment",
        "num_heads": int(num_heads if num_heads is not None else 0),
        "seq_len_q": M,
        "seq_len_k": N,
        "feature_dim": D,
        "n_examples": N_EXAMPLES,
        "canonical_warp": float(CANONICAL_WARP),
        "warp_sweep": [float(w) for w in WARP_SWEEP],
        "sweep": sweep,
        "baseline": baseline,
    }


# ----------------------------------------------------------------------
# Random reference model (smoke test only) — pure NumPy, no torch/GPU.
# ----------------------------------------------------------------------
def random_model_fn(num_heads: int = 4, seed: int = 0) -> ModelFn:
    """A ModelFn with the real signature returning random row-stochastic attn."""
    rng = np.random.default_rng(seed)

    def _fn(queries: np.ndarray, keys: np.ndarray) -> np.ndarray:
        m_ = int(np.asarray(queries).shape[0])
        n_ = int(np.asarray(keys).shape[0])
        logits = rng.standard_normal((num_heads, n_, m_))
        logits = logits - logits.max(axis=2, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=2, keepdims=True)

    return _fn


if __name__ == "__main__":
    pl = evaluate(random_model_fn())
    print("version:", pl["version"], "num_heads:", pl["num_heads"])
    print("sweep len:", len(pl["sweep"]), "baseline len:", len(pl["baseline"]))
    for s in pl["sweep"]:
        print(f"  warp={s['warp']}: best_head_overlap={s['best_head_overlap']:.4f}")
    print("smoke ok")
