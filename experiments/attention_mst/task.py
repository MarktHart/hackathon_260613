"""
task.py for attention_mst.

Owns the data generator and the evaluator for the goal. Two attempts at this
goal import this module instead of duplicating it, so they can never disagree
on what the data is or how it is scored.

Exports:
    Batch              — frozen dataclass, one evaluation instance.
    generate(seed)     — deterministic batch list for a seed.
    evaluate(model_fn) — runs model_fn (and an internal no-mechanism baseline)
                         over the canonical batches, returns the payload dict
                         consumed by benchmark.score().
    random_model_fn()  — a reference model_fn returning random scores of the
                         right shape; used by the smoke test.

Pure NumPy. No torch, no GPU, no I/O, no network.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

# ----------------------------------------------------------------------------
# Canonical configuration
# ----------------------------------------------------------------------------
N_HEADS: int = 12
NOISE_LEVELS: List[float] = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
N_SEEDS_PER_NOISE: int = 5
CANONICAL_NOISE: float = 0.5
EVAL_SEED: int = 42

# A model_fn maps observed (noisy) weights -> edge scores, both (N_HEADS, N_HEADS).
ModelFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Batch:
    """One evaluation instance: a planted graph + a noisy observation of it."""
    true_weights: np.ndarray        # (n_heads, n_heads) symmetric, zero diagonal
    noisy_weights: np.ndarray       # (n_heads, n_heads) symmetric, zero diagonal
    planted_mst_edges: np.ndarray   # (n_heads - 1, 2) int edges of the true MST
    planted_mst_weight: float       # total weight of the true MST
    noise_level: float              # std multiplier of the added noise


# ----------------------------------------------------------------------------
# Graph / MST helpers
# ----------------------------------------------------------------------------
def _generate_true_weights(n_heads: int, seed: int) -> np.ndarray:
    """Symmetric positive weights from LogNormal(0, 1), zero diagonal."""
    rng = np.random.default_rng(seed)
    n_edges = n_heads * (n_heads - 1) // 2
    vals = rng.lognormal(mean=0.0, sigma=1.0, size=n_edges)
    W = np.zeros((n_heads, n_heads), dtype=np.float64)
    idx = 0
    for i in range(n_heads):
        for j in range(i + 1, n_heads):
            W[i, j] = W[j, i] = vals[idx]
            idx += 1
    return W


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _kruskal_edges(order_weights: np.ndarray) -> np.ndarray:
    """
    Kruskal's MST over the upper triangle of `order_weights` (lower = preferred).
    Returns an (n-1, 2) int array of edges.
    """
    n = order_weights.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((order_weights[i, j], i, j))
    edges.sort(key=lambda e: e[0])

    dsu = _DSU(n)
    mst_edges: List[List[int]] = []
    for _, u, v in edges:
        if dsu.union(u, v):
            mst_edges.append([u, v])
            if len(mst_edges) == n - 1:
                break
    return np.array(mst_edges, dtype=int)


def _kruskal_mst(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """True MST: minimise total weight. Returns (edges, total_weight)."""
    edges = _kruskal_edges(weights)
    total = float(sum(weights[u, v] for u, v in edges))
    return edges, total


def _mst_edges_from_scores(scores: np.ndarray) -> np.ndarray:
    """Predicted MST edges, treating higher score as a more-preferred edge."""
    return _kruskal_edges(-scores)


def _edge_set(edges: np.ndarray) -> set:
    return {tuple(sorted(int(x) for x in e)) for e in edges}


def _prf1(pred_edges: np.ndarray, true_edges: np.ndarray) -> tuple[float, float, float]:
    """Precision, recall, F1 of edge recovery (unordered pairs)."""
    pred = _edge_set(pred_edges)
    true = _edge_set(true_edges)
    if not pred and not true:
        return 1.0, 1.0, 1.0
    common = len(pred & true)
    precision = common / len(pred) if pred else 0.0
    recall = common / len(true) if true else 0.0
    if precision + recall <= 0.0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _auroc_auprc(scores: np.ndarray, true_edges: np.ndarray) -> tuple[float, float]:
    """AUROC (Mann-Whitney U) and average-precision over upper-triangle edges."""
    n = scores.shape[0]
    true_set = _edge_set(true_edges)
    y_true: List[float] = []
    y_score: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            y_true.append(1.0 if (i, j) in true_set else 0.0)
            y_score.append(float(scores[i, j]))

    yt = np.asarray(y_true)
    ys = np.asarray(y_score)
    n_pos = int(yt.sum())
    n_neg = int((1 - yt).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.5
    if np.allclose(ys, ys[0]):
        return 0.5, n_pos / (n_pos + n_neg)

    # AUROC via rank-sum (average ranks to handle ties).
    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty(len(ys), dtype=np.float64)
    ranks[order] = np.arange(1, len(ys) + 1)
    # average tied ranks
    sorted_scores = ys[order]
    start = 0
    for end in range(1, len(ys) + 1):
        if end == len(ys) or sorted_scores[end] != sorted_scores[start]:
            if end - start > 1:
                avg = ranks[order[start:end]].mean()
                ranks[order[start:end]] = avg
            start = end
    sum_ranks_pos = ranks[yt == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2
    auroc = u / (n_pos * n_neg)

    # Average precision.
    desc = np.argsort(-ys, kind="mergesort")
    yt_sorted = yt[desc]
    tp = np.cumsum(yt_sorted)
    fp = np.cumsum(1 - yt_sorted)
    prec = tp / np.maximum(tp + fp, 1e-12)
    rec = tp / n_pos
    ap = 0.0
    prev_rec = 0.0
    for k in range(len(rec)):
        if yt_sorted[k] == 1.0:
            ap += (rec[k] - prev_rec) * prec[k]
            prev_rec = rec[k]
    return float(auroc), float(ap)


# ----------------------------------------------------------------------------
# Data generation
# ----------------------------------------------------------------------------
def generate(seed: int = 0) -> List[Batch]:
    """
    Deterministic: same seed -> identical batch list. Produces
    N_SEEDS_PER_NOISE instances at each noise level in NOISE_LEVELS.
    """
    batches: List[Batch] = []
    base_rng = np.random.default_rng(seed)
    for noise_level in NOISE_LEVELS:
        for _ in range(N_SEEDS_PER_NOISE):
            derived_seed = int(base_rng.integers(0, 2**31 - 1))
            true_weights = _generate_true_weights(N_HEADS, derived_seed)
            planted_edges, planted_weight = _kruskal_mst(true_weights)

            scale = float(np.median(true_weights[true_weights > 0]))
            noise = np.random.default_rng(derived_seed + 1_000_003).normal(
                0.0, noise_level * scale, size=true_weights.shape
            )
            noisy = true_weights + noise
            noisy = (noisy + noisy.T) / 2.0
            np.fill_diagonal(noisy, 0.0)
            noisy = np.maximum(noisy, 1e-8)

            batches.append(Batch(
                true_weights=true_weights,
                noisy_weights=noisy,
                planted_mst_edges=planted_edges,
                planted_mst_weight=planted_weight,
                noise_level=noise_level,
            ))
    return batches


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------
def _run_model(model_fn: ModelFn, by_noise: Dict[float, List[Batch]]) -> List[Dict[str, Any]]:
    """Run one model_fn over the batches, return one averaged record per noise level."""
    records: List[Dict[str, Any]] = []
    for noise_level in NOISE_LEVELS:
        batches_at_noise = by_noise[noise_level]
        f1s, precs, recs, aurocs, auprcs, wratios = [], [], [], [], [], []

        for batch in batches_at_noise:
            scores = np.asarray(model_fn(batch.noisy_weights), dtype=np.float64)
            if scores.shape != (N_HEADS, N_HEADS):
                raise ValueError(
                    f"model_fn returned shape {scores.shape}, expected ({N_HEADS}, {N_HEADS})"
                )
            scores = (scores + scores.T) / 2.0
            np.fill_diagonal(scores, 0.0)

            pred_edges = _mst_edges_from_scores(scores)
            pred_weight = float(sum(batch.true_weights[u, v] for u, v in pred_edges))

            precision, recall, f1 = _prf1(pred_edges, batch.planted_mst_edges)
            auroc, auprc = _auroc_auprc(scores, batch.planted_mst_edges)
            denom = batch.planted_mst_weight if batch.planted_mst_weight > 1e-12 else 1.0
            wratio = pred_weight / denom

            f1s.append(f1)
            precs.append(precision)
            recs.append(recall)
            aurocs.append(auroc)
            auprcs.append(auprc)
            wratios.append(wratio)

        records.append({
            "noise_level": float(noise_level),
            "edge_f1": float(np.mean(f1s)),
            "precision": float(np.mean(precs)),
            "recall": float(np.mean(recs)),
            "auroc": float(np.mean(aurocs)),
            "auprc": float(np.mean(auprcs)),
            "weight_ratio": float(np.mean(wratios)),
            "n_seeds": int(len(batches_at_noise)),
        })
    return records


def _baseline_model_fn(noisy_weights: np.ndarray) -> np.ndarray:
    """
    No-mechanism reference: score each edge by the negation of its observed
    (noisy) weight — i.e. run Kruskal directly on the noisy observation, with
    no denoising. Lower observed weight => higher score => preferred edge.
    """
    return -np.asarray(noisy_weights, dtype=np.float64)


def evaluate(model_fn: ModelFn) -> Dict[str, Any]:
    """
    Run `model_fn` (and the internal no-mechanism baseline) over the canonical
    batches and return the payload dict that benchmark.score() consumes.
    """
    batches = generate(seed=EVAL_SEED)
    by_noise: Dict[float, List[Batch]] = {}
    for b in batches:
        by_noise.setdefault(b.noise_level, []).append(b)

    sweep = _run_model(model_fn, by_noise)
    baseline = _run_model(_baseline_model_fn, by_noise)

    return {
        "version": 1,
        "model_name": "attention_mst",
        "n_heads": N_HEADS,
        "canonical_noise": CANONICAL_NOISE,
        "noise_levels": list(NOISE_LEVELS),
        "sweep": sweep,
        "baseline": baseline,
    }


def random_model_fn() -> ModelFn:
    """
    Reference model_fn returning random edge scores of the right shape. Pure
    NumPy, deterministic, no torch/GPU. Used by the pipeline smoke test.
    """
    def _fn(noisy_weights: np.ndarray) -> np.ndarray:
        n = int(np.asarray(noisy_weights).shape[0])
        rng = np.random.default_rng(12345)
        scores = rng.random((n, n))
        scores = (scores + scores.T) / 2.0
        np.fill_diagonal(scores, 0.0)
        return scores.astype(np.float64)
    return _fn
