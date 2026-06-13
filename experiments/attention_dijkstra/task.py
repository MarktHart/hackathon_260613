import heapq
from dataclasses import dataclass
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# model_fn contract (documented in README.md)
#
#   model_fn(weights: np.ndarray, source: int) -> np.ndarray
#
#   weights : (n, n) float64 adjacency matrix of an undirected, positively
#             weighted graph. weights[i, j] is the cost of edge (i, j), with
#             np.inf where no edge exists and 0.0 on the diagonal. The matrix
#             is symmetric.
#   source  : index of the single source node.
#   returns : (n,) float array of *predicted* shortest-path distances from
#             `source` to every node. predicted[source] should be ~0.
#
# The attempt's mechanism is whatever produces those predictions (iterated
# soft-min attention relaxation, a learned head, etc.). task.py only sees the
# numbers.
# ---------------------------------------------------------------------------
ModelFn = Callable[[np.ndarray, int], np.ndarray]

# Canonical measurement condition (see README.md).
N_NODES_SWEEP = [8, 16, 32, 64]
CANONICAL_N = 16
N_SEEDS = 20
EVAL_SEED = 42

# Edge weights sampled uniformly from [W_LOW, W_HIGH].
W_LOW = 1.0
W_HIGH = 10.0

# Accuracy tolerance: a node's predicted distance counts as correct when it is
# within REL_TOL of the true distance (plus a tiny absolute floor).
REL_TOL = 0.10
ABS_TOL = 1e-6


@dataclass(frozen=True)
class Batch:
    """One evaluation batch: a set of connected weighted graphs across seeds.

    Each entry i describes one (graph, source) problem instance:
        weights[i]   : (n, n) float64 symmetric adjacency with inf for missing
        sources[i]   : int source node
        n_nodes[i]   : int graph size (== weights[i].shape[0])
    """
    weights: list
    sources: list
    n_nodes: list


def _make_graph(rng: np.random.Generator, n: int) -> np.ndarray:
    """A connected, undirected, positively-weighted graph as a (n, n) matrix."""
    weights = np.full((n, n), np.inf, dtype=np.float64)
    np.fill_diagonal(weights, 0.0)

    def _add_edge(u: int, v: int) -> None:
        if u == v:
            return
        if not np.isfinite(weights[u, v]):
            w = float(rng.uniform(W_LOW, W_HIGH))
            weights[u, v] = w
            weights[v, u] = w

    # Random spanning tree guarantees connectivity.
    perm = rng.permutation(n)
    for i in range(1, n):
        u = int(perm[i])
        v = int(perm[int(rng.integers(0, i))])
        _add_edge(u, v)

    # Extra random edges (~n of them) to create alternative paths.
    for _ in range(n):
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        _add_edge(u, v)

    return weights


def _shortest_paths(weights: np.ndarray, source: int) -> np.ndarray:
    """Ground-truth single-source shortest paths via Dijkstra. Pure Python."""
    n = weights.shape[0]
    dist = np.full(n, np.inf, dtype=np.float64)
    dist[source] = 0.0
    visited = [False] * n
    pq = [(0.0, source)]
    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        row = weights[u]
        for v in range(n):
            w = row[v]
            if v != u and np.isfinite(w):
                nd = d + float(w)
                if nd < dist[v]:
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
    return dist


def generate(seed: int = 0) -> Batch:
    """Deterministic batch for the canonical sweep.

    Produces len(N_NODES_SWEEP) * N_SEEDS connected graphs. `seed` shifts the
    per-entry RNG so the whole batch is reproducible for a given seed.
    """
    weights_list = []
    sources = []
    n_nodes = []

    for ni, n in enumerate(N_NODES_SWEEP):
        for s in range(N_SEEDS):
            seed_i = (np.uint64(seed) * np.uint64(1_000_003)
                      + np.uint64(ni) * np.uint64(9_973)
                      + np.uint64(s)) & np.uint64(0xFFFFFFFF)
            r = np.random.default_rng(int(seed_i))

            weights = _make_graph(r, n)
            source = int(r.integers(0, n))

            weights_list.append(weights)
            sources.append(source)
            n_nodes.append(int(n))

    return Batch(weights=weights_list, sources=sources, n_nodes=n_nodes)


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) of the entries of `a`; ties share the mean rank."""
    a = np.asarray(a, dtype=np.float64)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(1, len(a) + 1, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(pred: np.ndarray, true: np.ndarray) -> float:
    """Spearman rank correlation in [-1, 1]; 0 when either side is constant."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    if pred.size < 2:
        return 0.0
    rp = _rankdata(pred)
    rt = _rankdata(true)
    sp = rp.std()
    st = rt.std()
    if sp < 1e-12 or st < 1e-12:
        return 0.0
    corr = float(np.mean((rp - rp.mean()) * (rt - rt.mean())) / (sp * st))
    return max(-1.0, min(1.0, corr))


def _accuracy(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of masked nodes whose predicted distance is within tolerance."""
    if not np.any(mask):
        return 0.0
    p = pred[mask]
    t = true[mask]
    ok = np.abs(p - t) <= (REL_TOL * np.abs(t) + ABS_TOL)
    return float(np.mean(ok))


def _onehop_baseline(weights: np.ndarray, source: int) -> np.ndarray:
    """No-propagation strawman: distance = direct edge weight, else infinity.

    This is shortest paths truncated to a single relaxation step — it has no
    mechanism for composing edges into multi-hop paths, so it is exactly right
    on direct neighbours and wrong (over-estimating) everywhere else.
    """
    pred = np.array(weights[source], dtype=np.float64)  # row = direct edges
    pred[source] = 0.0
    return pred


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch; return the benchmark payload."""
    batch = generate(seed=EVAL_SEED)

    method_by_n = {n: [] for n in N_NODES_SWEEP}
    base_by_n = {n: [] for n in N_NODES_SWEEP}

    for weights, source, n in zip(batch.weights, batch.sources, batch.n_nodes):
        true = _shortest_paths(weights, source)

        # Reachable, non-source nodes are the ones the mechanism must get right.
        mask = np.isfinite(true)
        mask[source] = False

        # --- Attempt's model ---
        pred = np.asarray(model_fn(weights, int(source)), dtype=np.float64).reshape(-1)
        if pred.shape != (n,):
            raise ValueError(
                f"model_fn returned shape {pred.shape}, expected ({n},)"
            )
        if np.any(mask) and not np.all(np.isfinite(pred[mask])):
            # Replace non-finite predictions on scored nodes with a large finite
            # value so metrics stay finite; such nodes simply read as wrong.
            pred = np.where(np.isfinite(pred), pred, 1e9)

        method_by_n[n].append({
            "distance_accuracy": _accuracy(pred, true, mask),
            "order_correlation": _spearman(pred[mask], true[mask]) if np.any(mask) else 0.0,
        })

        # --- One-hop baseline ---
        bpred = _onehop_baseline(weights, source)
        base_by_n[n].append({
            "distance_accuracy": _accuracy(bpred, true, mask),
            "order_correlation": _spearman(bpred[mask], true[mask]) if np.any(mask) else 0.0,
        })

    sweep = []
    linear_baseline = []
    for n in N_NODES_SWEEP:
        m = method_by_n[n]
        b = base_by_n[n]
        sweep.append({
            "n_nodes": int(n),
            "distance_accuracy": float(np.mean([r["distance_accuracy"] for r in m])),
            "order_correlation": float(np.mean([r["order_correlation"] for r in m])),
            "n_seeds": len(m),
        })
        linear_baseline.append({
            "n_nodes": int(n),
            "distance_accuracy": float(np.mean([r["distance_accuracy"] for r in b])),
            "order_correlation": float(np.mean([r["order_correlation"] for r in b])),
            "n_seeds": len(b),
        })

    return {
        "version": 1,
        "model_name": "synthetic_attention_dijkstra",
        "canonical_n": CANONICAL_N,
        "n_nodes_sweep": list(N_NODES_SWEEP),
        "rel_tol": REL_TOL,
        "sweep": sweep,
        "linear_baseline": linear_baseline,
    }


def random_model_fn() -> ModelFn:
    """A model_fn with the real signature whose body emits random distances.

    Pure NumPy; used by the pipeline smoke test.
    """
    rng = np.random.default_rng(0)

    def _random_fn(weights: np.ndarray, source: int) -> np.ndarray:
        n = np.asarray(weights).shape[0]
        out = rng.uniform(0.0, W_HIGH * n, size=n).astype(np.float64)
        out[source] = 0.0
        return out

    return _random_fn
