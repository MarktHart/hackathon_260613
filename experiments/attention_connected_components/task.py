"""Task for the `attention_connected_components` goal.

Synthetic graph generator + evaluator. Exports:

    generate(seed) -> Batch
    evaluate(model_fn) -> payload dict   (shape consumed by benchmark.score)
    random_model_fn() -> ModelFn         (pure-NumPy contract-shaped stub)

The model contract (`model_fn`):

    model_fn(adjacency: np.ndarray) -> np.ndarray

    adjacency : (N, N) float, symmetric, 0/1, zero diagonal. The undirected
                graph over N nodes.
    returns   : (N, N) float same-component *affinity*. affinity[i, j] is the
                model's belief that nodes i and j lie in the same connected
                component. The evaluator thresholds at 0.5; values may be any
                reals (only the >= 0.5 relation is read). The diagonal is
                ignored.

Recovering connected components is a *transitive closure* problem: a single
attention hop sees only direct neighbours, but two nodes in the same component
may be many hops apart. The adjacency-only baseline (affinity == adjacency)
therefore degrades as component diameter grows; a model that genuinely
computes the closure does not.
"""

from dataclasses import dataclass, field

import numpy as np

# Canonical measurement condition.
DIAMETERS = (1, 2, 3, 5)   # path diameter of each component in a slice
CANONICAL_DIAMETER = 3
NUM_COMPONENTS = 4         # K components per graph
NUM_GRAPHS = 8             # graphs averaged per diameter slice


@dataclass(frozen=True)
class Batch:
    """Deterministic fixture for the whole sweep.

    slices: tuple of (diameter, graphs) where graphs is a tuple of
            (adjacency, labels) pairs:
                adjacency : (N, N) float32 symmetric 0/1, zero diagonal
                labels    : (N,)  int32 connected-component id per node
    """
    slices: tuple = field(default_factory=tuple)
    diameters: tuple = DIAMETERS
    canonical_diameter: int = CANONICAL_DIAMETER
    num_components: int = NUM_COMPONENTS
    num_graphs: int = NUM_GRAPHS


def _build_graph(diameter: int, num_components: int, rng) -> tuple:
    """One graph: `num_components` disjoint paths, each of diameter `diameter`.

    A path of (diameter + 1) nodes has graph diameter == diameter. Node order
    is permuted so adjacency is not trivially block-structured.
    """
    comp_size = diameter + 1
    n = num_components * comp_size

    adjacency = np.zeros((n, n), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)

    perm = rng.permutation(n)
    idx = 0
    for c in range(num_components):
        nodes = perm[idx:idx + comp_size]
        idx += comp_size
        labels[nodes] = c
        # Chain the component's nodes into a path.
        for a, b in zip(nodes[:-1], nodes[1:]):
            adjacency[a, b] = 1.0
            adjacency[b, a] = 1.0
    return adjacency, labels


def generate(seed: int = 0) -> Batch:
    """Deterministic: same seed -> identical batch."""
    rng = np.random.default_rng(seed)
    slices = []
    for diameter in DIAMETERS:
        graphs = tuple(
            _build_graph(diameter, NUM_COMPONENTS, rng)
            for _ in range(NUM_GRAPHS)
        )
        slices.append((diameter, graphs))
    return Batch(slices=tuple(slices))


def _accumulate(counts: dict, pred: np.ndarray, truth: np.ndarray) -> None:
    """Tally a boolean same-component prediction vs truth over i<j pairs."""
    n = pred.shape[0]
    iu = np.triu_indices(n, k=1)
    p = pred[iu]
    t = truth[iu]
    counts["tp"] += int(np.count_nonzero(p & t))
    counts["fp"] += int(np.count_nonzero(p & ~t))
    counts["fn"] += int(np.count_nonzero(~p & t))
    counts["tn"] += int(np.count_nonzero(~p & ~t))


def evaluate(model_fn) -> dict:
    """Run `model_fn` over the canonical batch, return a benchmark-ready payload.

    For every diameter slice we accumulate a pairwise confusion matrix (over
    unordered node pairs) for both the model and the adjacency-only baseline.
    Aggregation into F1 / robustness happens in benchmark.score.
    """
    batch = generate(seed=0)

    sweep = []
    for diameter, graphs in batch.slices:
        model_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        base_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

        for adjacency, labels in graphs:
            affinity = np.asarray(model_fn(adjacency), dtype=np.float64)
            if affinity.shape != adjacency.shape:
                raise ValueError(
                    f"model_fn returned shape {affinity.shape}, "
                    f"expected {adjacency.shape}"
                )

            truth = labels[:, None] == labels[None, :]      # (N, N) bool
            model_pred = affinity >= 0.5                     # (N, N) bool
            base_pred = adjacency >= 0.5                     # 1-hop adjacency

            _accumulate(model_counts, model_pred, truth)
            _accumulate(base_counts, base_pred, truth)

        sweep.append({
            "diameter": int(diameter),
            "model": model_counts,
            "baseline": base_counts,
        })

    return {
        "version": 1,
        "canonical_diameter": int(batch.canonical_diameter),
        "num_components": int(batch.num_components),
        "num_graphs": int(batch.num_graphs),
        "sweep": sweep,
    }


def random_model_fn():
    """A contract-shaped stub: random affinities, pure NumPy, no torch/GPU.

    Signature and output shape match a real `model_fn` exactly so the pipeline
    smoke test exercises the full payload contract.
    """
    rng = np.random.default_rng(0)

    def _fn(adjacency: np.ndarray) -> np.ndarray:
        adjacency = np.asarray(adjacency)
        n = adjacency.shape[0]
        return rng.random((n, n))

    return _fn
