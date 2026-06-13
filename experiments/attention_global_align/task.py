"""Task: attention_global_align.

Data generator + evaluator for the "global alignment" goal. See README.md for
the full contract. Pure NumPy, deterministic, no I/O, no torch, no GPU.

The attempt hands `evaluate` a `model_fn(q, K) -> logits` (see ModelFn below);
`evaluate` runs it over the canonical sweep and returns a payload dict shaped
exactly as `benchmark.score` consumes it.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

# A model_fn maps a query vector q (d,) and a key matrix K (L, d) to a vector of
# attention *logits* over the L key positions. `evaluate` applies the softmax,
# so the logits need not be normalised and may be any real numbers.
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# Canonical measurement condition (see README.md). Changing any of these is a
# canonical-condition change and requires bumping benchmark.VERSION.
# ---------------------------------------------------------------------------
D = 32                     # residual / probe dimension
L = 12                     # number of key positions per sequence
N_SEQS = 24                # sequences ("seeds") sampled per sweep slice
DISTRACTOR_COS_SWEEP = [0.0, 0.25, 0.5, 0.75, 1.0]   # interference axis
CANONICAL_DISTRACTOR_COS = 0.5
EVAL_SEED = 7


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: N_SEQS * len(sweep) retrieval problems.

    Each entry i is a (query, keys, target, distractor) tuple:
      qs[i]            : (d,)   unit query vector
      Ks[i]            : (L, d) key matrix
      targets[i]       : int    index of the ground-truth key to retrieve
      distractors[i]   : int    index of the interfering key
      distractor_cos[i]: float  nominal cos(target_key, distractor_key)
    """
    qs: list[np.ndarray]
    Ks: list[np.ndarray]
    targets: list[int]
    distractors: list[int]
    distractor_cos: list[float]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def generate(seed: int = 0) -> Batch:
    """Deterministic batch for the canonical sweep.

    Produces len(DISTRACTOR_COS_SWEEP) * N_SEQS retrieval problems. `seed`
    shifts the per-entry RNG so the whole batch is reproducible for a seed.

    Construction: the query q is a random unit vector and the true target key
    is set equal to q (cosine 1), so a working retrieval head should place its
    mass there. A distractor key is placed at another position with a
    controlled cosine `c` to the target key; as c -> 1 the distractor becomes
    indistinguishable from the target and competes for attention. All other
    keys are random unit vectors.
    """
    qs: list[np.ndarray] = []
    Ks: list[np.ndarray] = []
    targets: list[int] = []
    distractors: list[int] = []
    cos_list: list[float] = []

    for ci, c in enumerate(DISTRACTOR_COS_SWEEP):
        for s in range(N_SEQS):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(ci) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))

            # Query and target key (cosine 1 to the query).
            q = _unit(r.normal(size=D))
            target_key = q.copy()

            # Pick distinct target / distractor positions.
            t = int(r.integers(0, L))
            dpos = int(r.integers(0, L - 1))
            if dpos >= t:
                dpos += 1

            # Distractor key with controlled cosine to the target key.
            ortho = r.normal(size=D)
            ortho = ortho - np.dot(ortho, target_key) * target_key
            ortho = _unit(ortho)
            distractor_key = _unit(c * target_key + np.sqrt(max(0.0, 1.0 - c * c)) * ortho)

            # Assemble the key matrix: random unit keys, then plant target/distractor.
            K = np.stack([_unit(r.normal(size=D)) for _ in range(L)], axis=0)
            K[t] = target_key
            K[dpos] = distractor_key

            qs.append(q.astype(np.float32))
            Ks.append(K.astype(np.float32))
            targets.append(t)
            distractors.append(dpos)
            cos_list.append(float(c))

    return Batch(qs=qs, Ks=Ks, targets=targets, distractors=distractors,
                 distractor_cos=cos_list)


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload.

    For each slice (distractor cosine) we average over N_SEQS sequences:
      global_alignment : mean attention mass on the true target position
      distractor_mass  : mean attention mass on the distractor position
      target_margin    : mean (target_mass - distractor_mass)
    A uniform-attention baseline (1/L on every key) is measured identically.
    """
    batch = generate(seed=EVAL_SEED)

    by_cos: dict[float, list[dict]] = {c: [] for c in DISTRACTOR_COS_SWEEP}
    base_by_cos: dict[float, list[float]] = {c: [] for c in DISTRACTOR_COS_SWEEP}

    for q, K, t, dpos, c in zip(
        batch.qs, batch.Ks, batch.targets, batch.distractors, batch.distractor_cos
    ):
        n_keys = K.shape[0]

        logits = np.asarray(model_fn(q, K), dtype=np.float64).reshape(-1)
        if logits.shape != (n_keys,):
            raise ValueError(
                f"model_fn returned shape {logits.shape}, expected ({n_keys},)"
            )
        if not np.all(np.isfinite(logits)):
            raise ValueError("model_fn returned non-finite logits")

        attn = _softmax(logits)
        target_mass = float(attn[t])
        distractor_mass = float(attn[dpos])

        by_cos[c].append({
            "global_alignment": target_mass,
            "distractor_mass": distractor_mass,
            "target_margin": target_mass - distractor_mass,
        })

        # Uniform baseline: mass on the target under 1/L attention.
        base_by_cos[c].append(1.0 / n_keys)

    sweep = []
    uniform_baseline = []
    for c in DISTRACTOR_COS_SWEEP:
        recs = by_cos[c]
        sweep.append({
            "distractor_cos": float(c),
            "global_alignment": float(np.mean([r["global_alignment"] for r in recs])),
            "distractor_mass": float(np.mean([r["distractor_mass"] for r in recs])),
            "target_margin": float(np.mean([r["target_margin"] for r in recs])),
            "n_seqs": len(recs),
        })
        base_vals = base_by_cos[c]
        uniform_baseline.append({
            "distractor_cos": float(c),
            "global_alignment": float(np.mean(base_vals)) if base_vals else 0.0,
            "n_seqs": len(base_vals),
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_global_align",
        "d": D,
        "seq_len": L,
        "canonical_distractor_cos": CANONICAL_DISTRACTOR_COS,
        "distractor_cos_sweep": list(DISTRACTOR_COS_SWEEP),
        "sweep": sweep,
        "uniform_baseline": uniform_baseline,
    }


def random_model_fn() -> ModelFn:
    """Return a ModelFn with the real signature emitting random logits.

    Pure NumPy; used by the pipeline smoke test. Called with NO arguments and
    returns a callable matching the documented `model_fn(q, K) -> logits`.
    """
    rng = np.random.default_rng(0)

    def _random_fn(q: np.ndarray, K: np.ndarray) -> np.ndarray:
        n_keys = np.asarray(K).shape[0]
        return rng.normal(size=n_keys).astype(np.float32)

    return _random_fn
