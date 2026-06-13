"""Task definition for the `attention_bfs` goal.

Question
--------
Can a mechanism propagate graph reachability one hop at a time the way
breadth-first search (BFS) does? Given a directed graph, a source node, and a
hop budget ``h``, the attempt must predict which nodes are reachable from the
source in at most ``h`` steps. A single application of attention can only move
information one hop; correctly answering for large ``h`` requires a genuine
multi-hop propagation mechanism, not a one-shot lookup.

This module owns the *data* and the *evaluation*. Every attempt imports it and
hands its ``model_fn`` to :func:`evaluate`; it never builds the payload itself.

Contracts
---------
- :func:`generate` is deterministic for a given seed.
- :func:`evaluate` takes one argument (the attempt's ``model_fn``) and returns
  a payload dict shaped exactly as ``benchmark.score`` consumes it.
- :func:`random_model_fn` returns a no-op ``model_fn`` of the correct signature
  used for the pipeline smoke test. Pure NumPy, no torch, no GPU.

`model_fn` signature
--------------------
    model_fn(adjacency: np.ndarray, source: int, hops: int) -> np.ndarray

    adjacency : (N, N) float/int array, ``adjacency[i, j] == 1`` iff there is a
                directed edge i -> j. No self loops.
    source    : int, the BFS source node index in [0, N).
    hops      : int, the hop budget h >= 1.
    returns   : (N,) float array of per-node reachability probabilities in
                [0, 1]. Entry k is P(node k reachable from source in <= hops
                steps). Thresholded at 0.5 by the evaluator.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Canonical measurement condition (fixed; do not change without bumping VERSION)
# ---------------------------------------------------------------------------
N_NODES = 24
N_GRAPHS = 48
EDGE_PROB = 0.10
HOPS_AXIS = (1, 2, 3, 4, 5)
CANONICAL_HOPS = 5
THRESHOLD = 0.5
TASK_VERSION = 1


@dataclass(frozen=True)
class Batch:
    """A fixed collection of directed graphs with BFS sources."""

    adjacency: list             # list of (N, N) np.ndarray, dtype float
    sources: list               # list of int
    n_nodes: int
    hops_axis: tuple = field(default=HOPS_AXIS)
    canonical_hops: int = CANONICAL_HOPS


def _bfs_distances(adj: np.ndarray, source: int) -> np.ndarray:
    """Shortest-path hop count from ``source``; unreachable -> +inf."""
    n = adj.shape[0]
    dist = np.full(n, np.inf, dtype=float)
    dist[source] = 0.0
    q = deque([source])
    while q:
        u = q.popleft()
        for v in np.nonzero(adj[u] > 0)[0]:
            if dist[v] == np.inf:
                dist[v] = dist[u] + 1.0
                q.append(int(v))
    return dist


def generate(seed: int = 0) -> Batch:
    """Deterministic: same seed -> same batch of graphs."""
    rng = np.random.RandomState(seed)
    adjacency: list = []
    sources: list = []
    for _ in range(N_GRAPHS):
        a = (rng.rand(N_NODES, N_NODES) < EDGE_PROB).astype(float)
        np.fill_diagonal(a, 0.0)
        adjacency.append(a)
        sources.append(int(rng.randint(N_NODES)))
    return Batch(
        adjacency=adjacency,
        sources=sources,
        n_nodes=N_NODES,
        hops_axis=HOPS_AXIS,
        canonical_hops=CANONICAL_HOPS,
    )


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _baseline_pred(adj: np.ndarray, source: int) -> np.ndarray:
    """No-mechanism strawman: nodes reachable in <= 1 hop (source + direct
    neighbours), regardless of the hop budget. One attention application."""
    n = adj.shape[0]
    pred = np.zeros(n, dtype=bool)
    pred[source] = True
    pred[np.nonzero(adj[source] > 0)[0]] = True
    return pred


def evaluate(model_fn) -> dict:
    """Run ``model_fn`` across the batch and every hop budget; return payload."""
    batch = generate(0)
    sweep = []
    for h in batch.hops_axis:
        m_tp = m_fp = m_fn = 0
        b_tp = b_fp = b_fn = 0
        m_correct = 0
        b_correct = 0
        total_nodes = 0
        for adj, src in zip(batch.adjacency, batch.sources):
            dist = _bfs_distances(adj, src)
            gt = dist <= h  # boolean (N,)

            probs = np.asarray(model_fn(adj, int(src), int(h)), dtype=float).reshape(-1)
            if probs.shape[0] != batch.n_nodes:
                raise ValueError(
                    f"model_fn returned shape {probs.shape}, "
                    f"expected ({batch.n_nodes},)"
                )
            pred = probs >= THRESHOLD

            base = _baseline_pred(adj, src)

            m_tp += int(np.sum(pred & gt))
            m_fp += int(np.sum(pred & ~gt))
            m_fn += int(np.sum(~pred & gt))
            m_correct += int(np.sum(pred == gt))

            b_tp += int(np.sum(base & gt))
            b_fp += int(np.sum(base & ~gt))
            b_fn += int(np.sum(~base & gt))
            b_correct += int(np.sum(base == gt))

            total_nodes += batch.n_nodes

        m_p, m_r, m_f1 = _prf1(m_tp, m_fp, m_fn)
        _, _, b_f1 = _prf1(b_tp, b_fp, b_fn)

        sweep.append(
            {
                "hops": int(h),
                "model_f1": float(m_f1),
                "model_acc": float(m_correct / total_nodes) if total_nodes else 0.0,
                "model_precision": float(m_p),
                "model_recall": float(m_r),
                "baseline_f1": float(b_f1),
                "baseline_acc": float(b_correct / total_nodes) if total_nodes else 0.0,
            }
        )

    return {
        "version": TASK_VERSION,
        "n_graphs": len(batch.adjacency),
        "n_nodes": batch.n_nodes,
        "edge_prob": EDGE_PROB,
        "canonical_hops": batch.canonical_hops,
        "hops_axis": list(batch.hops_axis),
        "threshold": THRESHOLD,
        "sweep": sweep,
    }


def random_model_fn():
    """A ``model_fn`` returning random reachability probabilities. Pure NumPy.

    Same signature as a real attempt's model function; used by the pipeline
    smoke test ``task.evaluate(task.random_model_fn())``."""
    rng = np.random.RandomState(12345)

    def fn(adjacency: np.ndarray, source: int, hops: int) -> np.ndarray:
        n = np.asarray(adjacency).shape[0]
        return rng.rand(n)

    return fn
