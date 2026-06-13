"""
attention_histogram — task: data generator + evaluator.

The goal asks: when a head must attend to ONE correct key among several
distractors, can the attempt's attention mechanism keep its attention
*histogram* sharp (low-entropy, single-peaked) and correctly targeted as the
distractor keys become more similar to the true target (interference rising)?

This file owns the data and the scoring-payload shape. Every attempt imports
`generate` / `evaluate` instead of duplicating them, so two attempts can never
disagree on what the data is.

Pure NumPy. No torch, no GPU, no I/O.
"""

from dataclasses import dataclass
from typing import Callable, List
import numpy as np

# A model_fn maps (query, keys) -> attention logits over the key positions.
#   query : (d,)            float32
#   keys  : (n_positions,d) float32
#   return: (n_positions,)  float   (pre-softmax logits)
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# Canonical measurement condition (documented in README.md).
# ---------------------------------------------------------------------------
D = 32                                  # residual / key dimensionality
N_POSITIONS = 16                        # number of key positions per query
KEY_SIM_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8]   # distractor↔target cosine (the axis)
CANONICAL_SIMILARITY = 0.0              # default condition: distinct keys
N_SEEDS = 16                            # conditions averaged per sweep point
QUERY_NOISE = 0.6                       # query corruption (makes it non-trivial)
EVAL_SEED = 7                           # seed used by evaluate()


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: many (query, keys, target_index) tuples.

    Each list has len(KEY_SIM_SWEEP) * N_SEEDS entries, aligned by index.
    """
    queries: List[np.ndarray]       # each (d,) float32
    keys: List[np.ndarray]          # each (n_positions, d) float32
    target_index: List[int]         # index of the correct key per entry
    similarities: List[float]       # nominal distractor cosine per entry


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def generate(seed: int = 0) -> Batch:
    """Deterministic batch for the canonical sweep.

    `seed` shifts the per-entry RNG so the whole batch is reproducible for a
    given seed. Same seed -> identical batch.
    """
    d = D
    n_positions = N_POSITIONS

    queries: List[np.ndarray] = []
    keys: List[np.ndarray] = []
    target_index: List[int] = []
    similarities: List[float] = []

    for ci, sim in enumerate(KEY_SIM_SWEEP):
        for s in range(N_SEEDS):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(ci) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))

            # Target key direction, on the unit sphere.
            t = _unit(r.normal(size=d))

            k = np.zeros((n_positions, d), dtype=np.float64)
            tgt = int(r.integers(n_positions))
            scale = np.sqrt(max(0.0, 1.0 - sim ** 2))
            for j in range(n_positions):
                if j == tgt:
                    k[j] = t
                else:
                    # Distractor with cosine ~= sim to the target direction.
                    o = r.normal(size=d)
                    o = o - np.dot(o, t) * t
                    o = _unit(o)
                    k[j] = _unit(sim * t + scale * o)

            # Query points at the target, corrupted by noise.
            q = _unit(t + QUERY_NOISE * r.normal(size=d))

            queries.append(q.astype(np.float32))
            keys.append(k.astype(np.float32))
            target_index.append(tgt)
            similarities.append(float(sim))

    return Batch(
        queries=queries,
        keys=keys,
        target_index=target_index,
        similarities=similarities,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(e, 1.0 / len(e))
    return e / s


def _concentration(attn: np.ndarray) -> float:
    """Histogram sharpness: 1 - H(attn)/log(n). Uniform->0, one-hot->1."""
    n = attn.shape[0]
    if n <= 1:
        return 1.0
    eps = 1e-12
    p = np.clip(attn, eps, 1.0)
    p = p / p.sum()
    H = float(-np.sum(p * np.log(p)))          # nats
    return float(max(0.0, min(1.0, 1.0 - H / np.log(n))))


def _entropy(attn: np.ndarray) -> float:
    """Shannon entropy of the attention histogram, in nats."""
    eps = 1e-12
    p = np.clip(attn, eps, 1.0)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch and return the scoring payload.

    The payload shape matches benchmark.score() exactly. Attempts never build
    the payload themselves — they hand `evaluate` a model_fn and get this back.
    """
    batch = generate(seed=EVAL_SEED)

    by_sim: dict = {c: [] for c in KEY_SIM_SWEEP}
    base_by_sim: dict = {c: [] for c in KEY_SIM_SWEEP}

    for q, k, tgt, sim in zip(
        batch.queries, batch.keys, batch.target_index, batch.similarities
    ):
        n_positions = k.shape[0]

        # --- Attempt's mechanism ---
        logits = np.asarray(model_fn(q, k), dtype=np.float64).reshape(-1)
        if logits.shape != (n_positions,):
            raise ValueError(
                f"model_fn returned shape {logits.shape}, "
                f"expected ({n_positions},)"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError("model_fn returned non-finite logits")
        attn = _softmax(logits)

        by_sim[sim].append({
            "sharpness": _concentration(attn),
            "entropy": _entropy(attn),
            "hit": 1.0 if int(np.argmax(attn)) == tgt else 0.0,
        })

        # --- Linear baseline: plain dot-product attention (no mechanism) ---
        base_logits = k.astype(np.float64) @ q.astype(np.float64)
        base_attn = _softmax(base_logits)
        base_by_sim[sim].append({
            "sharpness": _concentration(base_attn),
            "hit": 1.0 if int(np.argmax(base_attn)) == tgt else 0.0,
        })

    sweep = []
    linear_baseline = []
    for sim in KEY_SIM_SWEEP:
        recs = by_sim[sim]
        sweep.append({
            "similarity": float(sim),
            "attention_sharpness": float(np.mean([r["sharpness"] for r in recs])),
            "attention_entropy": float(np.mean([r["entropy"] for r in recs])),
            "target_hit_rate": float(np.mean([r["hit"] for r in recs])),
            "n_seeds": len(recs),
        })
        brecs = base_by_sim[sim]
        linear_baseline.append({
            "similarity": float(sim),
            "attention_sharpness": float(np.mean([r["sharpness"] for r in brecs])),
            "target_hit_rate": float(np.mean([r["hit"] for r in brecs])),
            "n_seeds": len(brecs),
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_histogram",
        "d": D,
        "n_positions": N_POSITIONS,
        "chance_hit_rate": 1.0 / N_POSITIONS,
        "canonical_similarity": CANONICAL_SIMILARITY,
        "key_sim_sweep": list(KEY_SIM_SWEEP),
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


def random_model_fn() -> ModelFn:
    """Return a model_fn with the real signature whose body emits random logits.

    Pure NumPy. Used by the pipeline smoke test:
        payload = task.evaluate(task.random_model_fn())
    """
    rng = np.random.default_rng(0)

    def _random_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        n_positions = np.asarray(keys).shape[0]
        return rng.normal(size=n_positions).astype(np.float32)

    return _random_fn
