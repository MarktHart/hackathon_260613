"""Synthetic graph-coloring task for attention_graph_color.

Generates random graphs with a known proper coloring and evaluates a model's
attention matrix against the coloring structure. A model that "understands"
proper colorings should place more attention on differently-coloured node
pairs (and especially on the edges that connect different colours) than on
same-coloured pairs.

Pure NumPy. No I/O, no network, no torch.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np

# --- Canonical measurement condition (see README.md) ---
N_VALUES = [20, 40, 60]
P_VALUES = [0.1, 0.2, 0.3]
GRAPHS_PER_COMBO = 5          # 3 * 3 * 5 = 45 graphs
CANONICAL_N = 40
CANONICAL_P = 0.2
EVAL_SEED = 0


ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Model function signature.

Args:
    adj:   (n, n) symmetric adjacency matrix, float32, entries in {0, 1},
           zero diagonal.
    feats: (n, k+1) node features, float32: one-hot colour (k dims) plus a
           normalised degree scalar.

Returns:
    (n, n) attention-weight matrix, float32. Rows for non-isolated nodes
    should be non-negative; rows for isolated nodes (degree 0) may be zeros.
    Normalisation is not required — the evaluator only reads relative mass.
"""


@dataclass(frozen=True)
class Batch:
    """A set of graphs with proper colorings."""
    adjacency: list[np.ndarray]          # list of (n, n) float32, symmetric
    features: list[np.ndarray]           # list of (n, k+1) float32
    colorings: list[np.ndarray]          # list of (n,) int32 in [0, k-1]
    num_colors_list: list[int]           # k for each graph
    n_list: list[int]                    # n for each graph


def _greedy_coloring(adj: np.ndarray) -> tuple[np.ndarray, int]:
    """Return a proper greedy coloring and the number of colours used.

    Nodes are processed in descending-degree order for better colour economy.
    Guaranteed proper: a node never receives a colour used by a neighbour.
    """
    n = adj.shape[0]
    degrees = adj.sum(axis=1).astype(int)
    order = np.argsort(-degrees)  # highest degree first
    colors = np.full(n, -1, dtype=np.int32)
    max_color = -1

    for u in order:
        neighbor_colors = {
            int(colors[v]) for v in np.where(adj[u] > 0)[0] if colors[v] != -1
        }
        c = 0
        while c in neighbor_colors:
            c += 1
        colors[u] = c
        if c > max_color:
            max_color = c

    return colors, max_color + 1


def _build_features(colors: np.ndarray, k: int, adj: np.ndarray) -> np.ndarray:
    """Build (n, k+1) features: one-hot colour + normalised degree."""
    n = colors.shape[0]
    feats = np.zeros((n, k + 1), dtype=np.float32)
    feats[np.arange(n), colors] = 1.0
    degrees = adj.sum(axis=1).astype(np.float32)
    feats[:, -1] = degrees / max(1.0, n - 1)
    return feats


def generate(seed: int = 0) -> Batch:
    """Generate a deterministic batch of graphs with proper colorings.

    Same seed -> identical batch. The sweep covers n in {20, 40, 60} and
    p in {0.1, 0.2, 0.3}, 5 graphs each (45 graphs total). The canonical
    slice is n=40 (all p).
    """
    rng = np.random.default_rng(seed)

    adjacency: list[np.ndarray] = []
    features: list[np.ndarray] = []
    colorings: list[np.ndarray] = []
    num_colors_list: list[int] = []
    n_list: list[int] = []

    for n in N_VALUES:
        for p in P_VALUES:
            for _ in range(GRAPHS_PER_COMBO):
                # Erdős–Rényi G(n, p), symmetric, zero diagonal.
                upper = rng.random((n, n)) < p
                mask = np.triu(upper, k=1)
                mask = mask | mask.T
                adj = mask.astype(np.float32)

                colors, k = _greedy_coloring(adj)
                feats = _build_features(colors, k, adj)

                adjacency.append(adj)
                features.append(feats)
                colorings.append(colors)
                num_colors_list.append(int(k))
                n_list.append(int(n))

    return Batch(
        adjacency=adjacency,
        features=features,
        colorings=colorings,
        num_colors_list=num_colors_list,
        n_list=n_list,
    )


def _attention_stats(attn: np.ndarray, adj: np.ndarray, colors: np.ndarray) -> dict:
    """Reduce one (n, n) attention matrix to scalar coloring statistics."""
    n = adj.shape[0]
    triu = np.triu(np.ones((n, n), dtype=bool), k=1)
    edge_mask = (adj > 0) & triu

    color_eq = colors[:, None] == colors[None, :]
    same_mask = color_eq & triu
    diff_mask = (~color_eq) & triu

    same_edge = same_mask & edge_mask
    diff_edge = diff_mask & edge_mask

    def _mean(mask: np.ndarray) -> float:
        return float(attn[mask].mean()) if mask.any() else 0.0

    degrees = adj.sum(axis=1)
    return {
        "same_color_attention": _mean(same_mask),
        "diff_color_attention": _mean(diff_mask),
        "cross_edge_same_color": _mean(same_edge),
        "cross_edge_diff_color": _mean(diff_edge),
        "isolated_node_fraction": float((degrees == 0).sum()) / max(1, n),
        "edge_density": float(edge_mask.sum()) / max(1.0, n * (n - 1) / 2.0),
    }


def _uniform_attention(n: int, adj: np.ndarray) -> np.ndarray:
    """No-mechanism baseline: uniform attention over all other nodes.

    Isolated nodes (degree 0) still attend uniformly so that pair statistics
    have a non-trivial reference; this is the structureless strawman.
    """
    attn = np.full((n, n), 1.0 / max(1, n - 1), dtype=np.float32)
    np.fill_diagonal(attn, 0.0)
    return attn


def evaluate(model_fn: ModelFn) -> dict:
    """Run `model_fn` over the canonical batch and return the payload dict
    that benchmark.score consumes exactly.
    """
    batch = generate(seed=EVAL_SEED)

    sweep: list[dict] = []
    baseline_sweep: list[dict] = []

    for idx, (adj, feats, colors, k, n) in enumerate(zip(
        batch.adjacency, batch.features, batch.colorings,
        batch.num_colors_list, batch.n_list,
    )):
        attn = np.asarray(model_fn(adj, feats))
        if attn.shape != adj.shape:
            raise ValueError(
                f"model_fn returned shape {attn.shape}, expected {adj.shape} "
                f"for graph {idx}"
            )
        attn = attn.astype(np.float64, copy=False)

        stats = _attention_stats(attn, adj, colors)
        stats["graph_idx"] = idx
        stats["num_nodes"] = int(n)
        stats["num_colors"] = int(k)
        sweep.append(stats)

        base_attn = _uniform_attention(n, adj).astype(np.float64)
        bstats = _attention_stats(base_attn, adj, colors)
        bstats["graph_idx"] = idx
        bstats["num_nodes"] = int(n)
        baseline_sweep.append(bstats)

    return {
        "version": 1,
        "model_name": "synthetic_attention_graph_color",
        "canonical_n": CANONICAL_N,
        "canonical_p": CANONICAL_P,
        "n_values": list(N_VALUES),
        "num_graphs": len(sweep),
        "sweep": sweep,
        "baseline_sweep": baseline_sweep,
    }


def random_model_fn() -> ModelFn:
    """Return a model_fn with the real signature whose body emits random,
    non-negative attention rows. Pure NumPy; used by the pipeline smoke test.
    """
    rng = np.random.default_rng(0)

    def _fn(adj: np.ndarray, feats: np.ndarray) -> np.ndarray:
        n = adj.shape[0]
        attn = rng.random((n, n)).astype(np.float32)
        np.fill_diagonal(attn, 0.0)
        row_sums = attn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        return (attn / row_sums).astype(np.float32)

    return _fn
