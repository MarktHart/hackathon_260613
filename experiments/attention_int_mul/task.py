"""Task for the attention_int_mul goal.

Synthetic integer-multiplication *routing*: a query encodes two operands
(a, b); a row of candidate positions each encode an integer value; the head
must attend to the single position whose value equals the product a * b.

See README.md for the full contract. Pure NumPy, no torch, no GPU — the
pipeline smoke test runs `evaluate(random_model_fn())` then `benchmark.score`.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

# ---------------------------------------------------------------------------
# model_fn signature (the goal's contract with attempts; see README.md):
#
#   model_fn(a_vec, b_vec, key_vecs) -> attn_logits
#       a_vec:    (d,)               embedding phi(a) of operand a
#       b_vec:    (d,)               embedding phi(b) of operand b
#       key_vecs: (n_positions, d)   embeddings phi(value_i) of each candidate
#       returns:  (n_positions,)     unnormalised attention logits
# ---------------------------------------------------------------------------
ModelFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]

# ---------------------------------------------------------------------------
# Canonical measurement condition (see README.md).
# ---------------------------------------------------------------------------
D = 128                       # embedding dimension
V = 1024                      # integer-embedding table covers [0, V-1]
N_POSITIONS = 16              # candidate positions per trial
K_SWEEP = [2, 4, 8, 16, 32]   # operand range: a, b in [0, K-1]
CANONICAL_K = 8               # canonical difficulty slice
N_TRIALS = 200                # trials per K, averaged
EVAL_SEED = 42                # fixed seed used by evaluate()
EMBED_SEED = 12345            # fixed seed for the (stable) integer embedding

# Largest product the sweep can produce: (max K - 1) ** 2 must fit in [0, V-1].
assert (max(K_SWEEP) - 1) ** 2 < V, "embedding table too small for sweep"

# ---------------------------------------------------------------------------
# Fixed integer embedding table phi: {0..V-1} -> unit vectors in R^d.
# Stable across all seeds so attempts can learn / reconstruct the map.
# ---------------------------------------------------------------------------
def _build_embed_table() -> np.ndarray:
    rng = np.random.default_rng(EMBED_SEED)
    table = rng.normal(size=(V, D)).astype(np.float64)
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return table / norms


INT_EMBED = _build_embed_table()  # (V, d)


def embed(n: int | np.ndarray) -> np.ndarray:
    """phi(n): look up the fixed unit embedding of integer n (or an array)."""
    return INT_EMBED[np.asarray(n, dtype=np.int64)]


@dataclass(frozen=True)
class Batch:
    """One evaluation batch, flattened over (K, trial).

    Each row i is a routing trial: operands a[i], b[i], product[i] = a*b,
    candidate integer values candidates[i] (one of which is the product at
    column true_index[i]).
    """
    k: np.ndarray            # (T,)              operand range for the trial
    a: np.ndarray            # (T,)              operand a
    b: np.ndarray            # (T,)              operand b
    product: np.ndarray      # (T,)             ground-truth product a*b
    candidates: np.ndarray   # (T, n_positions)  candidate integer values
    true_index: np.ndarray   # (T,)              column holding the product


def _valid_products(k: int) -> np.ndarray:
    """Distinct products a*b for a, b in [0, k-1] — the 'confusable' pool."""
    vals = {a * b for a in range(k) for b in range(k)}
    return np.array(sorted(vals), dtype=np.int64)


def generate(seed: int = 0) -> Batch:
    """Deterministic batch for the canonical sweep.

    Returns len(K_SWEEP) * N_TRIALS routing trials. Distractors are drawn
    preferentially from other valid products of the same K (so they are
    confusable), padded with uniform integers from [0, V-1] when a small K
    does not offer enough distinct products.
    """
    rng = np.random.default_rng(seed)

    ks, as_, bs, prods, cands, tidx = [], [], [], [], [], []

    for k in K_SWEEP:
        pool = _valid_products(k)
        for _ in range(N_TRIALS):
            a = int(rng.integers(0, k))
            b = int(rng.integers(0, k))
            p = a * b

            # Distractor pool: other valid products of this K.
            distract_pool = pool[pool != p]
            n_need = N_POSITIONS - 1

            if distract_pool.size >= n_need:
                chosen = rng.choice(distract_pool, size=n_need, replace=False)
                distractors = list(int(x) for x in chosen)
            else:
                distractors = [int(x) for x in distract_pool]
                # Pad with uniform integers in [0, V-1], distinct from all used.
                used = set(distractors) | {p}
                while len(distractors) < n_need:
                    cand = int(rng.integers(0, V))
                    if cand not in used:
                        used.add(cand)
                        distractors.append(cand)

            values = np.array([p] + distractors, dtype=np.int64)
            perm = rng.permutation(N_POSITIONS)
            values = values[perm]
            true_index = int(np.where(perm == 0)[0][0])

            ks.append(k)
            as_.append(a)
            bs.append(b)
            prods.append(p)
            cands.append(values)
            tidx.append(true_index)

    return Batch(
        k=np.array(ks, dtype=np.int64),
        a=np.array(as_, dtype=np.int64),
        b=np.array(bs, dtype=np.int64),
        product=np.array(prods, dtype=np.int64),
        candidates=np.stack(cands).astype(np.int64),
        true_index=np.array(tidx, dtype=np.int64),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    s = e.sum()
    if s <= 0 or not np.isfinite(s):
        return np.full_like(e, 1.0 / e.size)
    return e / s


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload.

    For each trial the attempt's logits are softmaxed; routing is correct when
    argmax lands on the product position. The linear baseline routes additively
    (score_i = key_i . (phi(a) + phi(b))), which matches a *sum* direction
    rather than a *product* and is the no-mechanism reference.
    """
    batch = generate(seed=EVAL_SEED)
    T = batch.k.shape[0]

    # Per-K accumulators.
    acc_sum: dict[int, float] = {k: 0.0 for k in K_SWEEP}
    mass_sum: dict[int, float] = {k: 0.0 for k in K_SWEEP}
    base_acc_sum: dict[int, float] = {k: 0.0 for k in K_SWEEP}
    count: dict[int, int] = {k: 0 for k in K_SWEEP}

    for i in range(T):
        k = int(batch.k[i])
        a_vec = embed(batch.a[i])                 # (d,)
        b_vec = embed(batch.b[i])                 # (d,)
        key_vecs = embed(batch.candidates[i])     # (n_positions, d)
        true_idx = int(batch.true_index[i])
        n_pos = key_vecs.shape[0]

        # --- Attempt's model ---
        logits = np.asarray(model_fn(a_vec, b_vec, key_vecs), dtype=np.float64).reshape(-1)
        if logits.shape != (n_pos,):
            raise ValueError(
                f"model_fn returned shape {logits.shape}, expected ({n_pos},)"
            )
        if not np.all(np.isfinite(logits)):
            # Treat non-finite logits as a uniform (chance) prediction.
            logits = np.zeros(n_pos, dtype=np.float64)
        probs = _softmax(logits)
        pred = int(np.argmax(probs))

        acc_sum[k] += 1.0 if pred == true_idx else 0.0
        mass_sum[k] += float(probs[true_idx])

        # --- Linear (additive) baseline ---
        base_scores = key_vecs @ (a_vec + b_vec)
        base_pred = int(np.argmax(base_scores))
        base_acc_sum[k] += 1.0 if base_pred == true_idx else 0.0

        count[k] += 1

    sweep = []
    linear_baseline = []
    for k in K_SWEEP:
        n = max(count[k], 1)
        sweep.append({
            "k": int(k),
            "routing_accuracy": acc_sum[k] / n,
            "attended_mass": mass_sum[k] / n,
            "n_trials": count[k],
        })
        linear_baseline.append({
            "k": int(k),
            "routing_accuracy": base_acc_sum[k] / n,
            "n_trials": count[k],
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_int_mul",
        "d": D,
        "n_positions": N_POSITIONS,
        "canonical_k": CANONICAL_K,
        "k_sweep": list(K_SWEEP),
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


def random_model_fn() -> ModelFn:
    """Reference model_fn (real signature) emitting random logits.

    Pure NumPy; used by the pipeline smoke test. Returns chance-level routing.
    """
    rng = np.random.default_rng(0)

    def _random_fn(a_vec: np.ndarray, b_vec: np.ndarray, key_vecs: np.ndarray) -> np.ndarray:
        n_pos = np.asarray(key_vecs).shape[0]
        return rng.normal(size=n_pos).astype(np.float64)

    return _random_fn
