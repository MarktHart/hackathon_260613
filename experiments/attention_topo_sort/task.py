"""Task: does an attention matrix encode a DAG's topological (partial) order?

Exports:
    generate(seed) -> Batch          deterministic synthetic DAGs
    evaluate(model_fn) -> payload     runs model_fn, returns benchmark payload
    random_model_fn() -> ModelFn      a random/zero baseline with the real signature

Pure NumPy. No torch, no GPU, no I/O.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# ModelFn signature as documented in README.md:
#   model_fn(adjacency: (n, n) array, n: int) -> (n, n) attention matrix
ModelFn = Callable[[np.ndarray, int], np.ndarray]

# --- Canonical measurement condition (see README.md) ---
N_NODES = 8
N_DAGS = 24
DENSITY_SWEEP = [0.1, 0.2, 0.3, 0.5]
CANONICAL_DENSITY = 0.3
EVAL_SEED = 0


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: a tuple of DAG adjacency matrices per density."""
    n_nodes: int
    n_dags: int
    densities: tuple[float, ...]
    canonical_density: float
    # dags[i] is the tuple of (n, n) bool adjacency matrices at densities[i]
    dags: tuple[tuple[np.ndarray, ...], ...]


def _sample_dag(rng: np.random.Generator, n: int, density: float) -> np.ndarray:
    """Sample an acyclic adjacency matrix.

    Edges only go from earlier to later position in a random permutation, which
    guarantees acyclicity. adjacency[i, j] == True means edge i -> j.
    """
    perm = rng.permutation(n)
    adj = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < density:
                i, j = int(perm[a]), int(perm[b])  # perm[a] precedes perm[b]
                adj[i, j] = True
    return adj


def generate(seed: int = 0) -> Batch:
    """Deterministic batch of DAGs for the canonical sweep.

    Same `seed` -> identical DAGs.
    """
    dags: list[tuple[np.ndarray, ...]] = []
    for di, density in enumerate(DENSITY_SWEEP):
        per_density: list[np.ndarray] = []
        for k in range(N_DAGS):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(di) * np.uint64(9_973)
                      + np.uint64(k)) & np.uint64(0xFFFFFFFF)
            rng = np.random.default_rng(int(seed_i))
            per_density.append(_sample_dag(rng, N_NODES, density))
        dags.append(tuple(per_density))

    return Batch(
        n_nodes=N_NODES,
        n_dags=N_DAGS,
        densities=tuple(DENSITY_SWEEP),
        canonical_density=CANONICAL_DENSITY,
        dags=tuple(dags),
    )


def _ancestors(adj: np.ndarray) -> np.ndarray:
    """Transitive closure: reach[a, d] == True iff a is an ancestor of d.

    Floyd-Warshall style boolean closure. Diagonal stays False (no self-edges,
    no cycles).
    """
    reach = np.asarray(adj, dtype=bool).copy()
    n = reach.shape[0]
    for k in range(n):
        # a can reach d if a already reaches d, OR a reaches k and k reaches d
        reach |= reach[:, k:k + 1] & reach[k:k + 1, :]
    return reach


def _normalize_rows(m: np.ndarray, n: int) -> np.ndarray:
    """Defensive row-stochastic normalisation of an attention matrix.

    Coerces to float, scrubs NaN/inf, clips negatives, renormalises rows.
    A degenerate all-zero row becomes uniform (so it contributes only ties).
    """
    m = np.asarray(m, dtype=float)
    if m.shape != (n, n):
        raise ValueError(
            f"model_fn returned shape {m.shape}, expected {(n, n)}"
        )
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    m = np.clip(m, 0.0, None)
    s = m.sum(axis=1, keepdims=True)
    zero_rows = (s[:, 0] == 0.0)
    s[s == 0.0] = 1.0
    out = m / s
    if zero_rows.any():
        out[zero_rows] = 1.0 / n
    return out


def _topo_respect(attn: np.ndarray, anc: np.ndarray) -> tuple[float, int]:
    """Fraction of ordered ancestor pairs respected by the attention matrix.

    For each (a, d) with a an ancestor of d, credit 1 if attn[d, a] > attn[a, d],
    0.5 on a tie, 0 otherwise. Returns (sum_of_credit, n_pairs).
    """
    a_idx, d_idx = np.where(anc)
    if a_idx.size == 0:
        return 0.0, 0
    back = attn[d_idx, a_idx]   # descendant attends to ancestor
    fwd = attn[a_idx, d_idx]    # ancestor attends to descendant
    credit = np.where(back > fwd, 1.0, np.where(back == fwd, 0.5, 0.0))
    return float(credit.sum()), int(a_idx.size)


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch, return the benchmark payload."""
    batch = generate(EVAL_SEED)

    sweep: list[dict] = []
    for di, density in enumerate(batch.densities):
        total_credit = 0.0
        total_pairs = 0
        for adj in batch.dags[di]:
            n = adj.shape[0]
            anc = _ancestors(adj)
            attn = _normalize_rows(model_fn(adj.copy(), n), n)
            credit, pairs = _topo_respect(attn, anc)
            total_credit += credit
            total_pairs += pairs

        respect = (total_credit / total_pairs) if total_pairs else 0.0
        # Uniform attention -> every pair is an exact tie -> 0.5 each.
        uniform = 0.5
        sweep.append({
            "density": float(density),
            "topo_respect": float(respect),
            "uniform_respect": float(uniform),
            "pairs": int(total_pairs),
        })

    return {
        "canonical_density": float(batch.canonical_density),
        "n_nodes": int(batch.n_nodes),
        "n_dags": int(batch.n_dags),
        "model_name": "attempt",
        "sweep": sweep,
    }


def random_model_fn() -> ModelFn:
    """A random baseline with the exact real `model_fn` signature.

    Returns non-negative random attention weights of shape (n, n). Pure NumPy.
    Expected to score ~0.5 topo_respect (chance).
    """
    rng = np.random.default_rng(12345)

    def model_fn(adjacency: np.ndarray, n: int) -> np.ndarray:
        return rng.random((n, n))

    return model_fn
