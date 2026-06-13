"""attention_topo_sort / pass_3 — hand-built level-biased attention.

Hypothesis
----------
A genuine single-head attention layer (bilinear QK score + softmax) can encode
the *partial order* of a DAG — not by recomputing the evaluator's transitive
closure, but by giving each node a scalar **topological level** (longest path
from a source, computed by iterated max message-passing over the adjacency) and
letting the attention score be a bilinear interaction of those levels:

    score[i, j] = -beta * level[i] * level[j]
    attn        = softmax(score, dim=keys)

Because every ancestor `a` of a descendant `d` has level[a] < level[d]
*by construction* of the longest-path level, the softmax places strictly more
mass on the lower-level (earlier) node from the higher-level node's row than
vice-versa, so attn[d, a] > attn[a, d] for every ordered ancestor pair. The
topological sort literally falls out of the attention's level ordering.

This file:
  * registers `level_attention_fn` as the benchmarked model_fn (-> benchmark.json)
  * runs an ABLATION (beta=0 -> uniform) and a STRAWMAN (direct-edge-only
    attention) through the *same* task.evaluate, saving the comparison
  * sweeps node count N from 4..64 (2 orders of magnitude over edge count) to
    map the operating range
  * dumps canonical per-DAG (adjacency, level, attention) artefacts for the viz

Everything that matters runs in torch on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
BETA = 2.0


# --------------------------------------------------------------------------- #
# The mechanism (all GPU torch)                                               #
# --------------------------------------------------------------------------- #
def _compute_levels(adj_t: torch.Tensor, n: int) -> torch.Tensor:
    """Longest-path topological level per node, via iterated max message-passing.

    adj_t[i, j] == 1 means edge i -> j, so the predecessors of node i are the
    nodes j with adj_t[j, i] == 1. level[i] = 0 for sources, otherwise
    1 + max level over predecessors. Converges in <= n-1 hops; we run n.
    """
    level = torch.zeros(n, device=DEVICE)
    for _ in range(n):
        # contrib[j, i] = adj[j, i] * (level[j] + 1)  (0 where there is no edge)
        contrib = adj_t * (level.unsqueeze(1) + 1.0)
        new_level = contrib.max(dim=0).values            # max over predecessors j
        level = torch.clamp(new_level, min=0.0)
    return level


def level_attention_fn(adjacency: np.ndarray, n: int) -> np.ndarray:
    """Hand-built bilinear attention biased toward topologically-early nodes."""
    adj_t = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    level = _compute_levels(adj_t, n)
    # Scale-invariant so the bilinear magnitude never blows up the softmax at
    # large N (raw level*level would underflow exp in float32 for deep DAGs).
    # Strict level ordering of ancestor < descendant is preserved.
    scale = level / torch.clamp(level.max(), min=1.0)          # in [0, 1]
    # Genuine query-dependent (bilinear) attention score.
    scores = -BETA * scale.unsqueeze(1) * scale.unsqueeze(0)   # [i, j]
    attn = torch.softmax(scores, dim=1)
    return attn.detach().cpu().numpy()


def direct_edge_attention_fn(adjacency: np.ndarray, n: int) -> np.ndarray:
    """STRAWMAN: attend only to *direct* predecessors (no transitive closure).

    Captures every direct edge but is blind to multi-hop ancestry, so it ties
    (credit 0.5) on every transitive-only ancestor pair.
    """
    adj_t = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    pred = adj_t.t()                       # pred[i, j] = 1 iff j is a direct predecessor of i
    scores = 4.0 * pred
    attn = torch.softmax(scores, dim=1)
    return attn.detach().cpu().numpy()


def uniform_ablation_fn(adjacency: np.ndarray, n: int) -> np.ndarray:
    """ABLATION: knock out the level bias (beta=0). Attention collapses to
    uniform -> every ancestor pair is a tie -> topo_respect == 0.5."""
    adj_t = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    scores = 0.0 * adj_t                   # all zeros, but on GPU
    attn = torch.softmax(scores, dim=1)
    return attn.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Scoring helper that reuses the task's own metric internals                  #
# --------------------------------------------------------------------------- #
def _score_on_dags(task, dags, model_fn) -> float:
    total_credit, total_pairs = 0.0, 0
    for adj in dags:
        n = adj.shape[0]
        anc = task._ancestors(adj)
        attn = task._normalize_rows(model_fn(adj.copy(), n), n)
        credit, pairs = task._topo_respect(attn, anc)
        total_credit += credit
        total_pairs += pairs
    return (total_credit / total_pairs) if total_pairs else 0.0


def main() -> None:
    import json

    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # 1) Headline benchmark — the level-attention mechanism.
    payload = task.evaluate(level_attention_fn)
    payload["model_name"] = "level_biased_bilinear_attention (hand-built)"
    record_benchmark(__file__, run_dir, payload)

    # 2) Method comparison through the *same* evaluator: level vs strawman vs ablation.
    cmp_level = task.evaluate(level_attention_fn)
    cmp_direct = task.evaluate(direct_edge_attention_fn)
    cmp_uniform = task.evaluate(uniform_ablation_fn)
    densities = [r["density"] for r in cmp_level["sweep"]]
    comparison = {
        "densities": densities,
        "level_attention": [r["topo_respect"] for r in cmp_level["sweep"]],
        "direct_edge_strawman": [r["topo_respect"] for r in cmp_direct["sweep"]],
        "uniform_ablation": [r["topo_respect"] for r in cmp_uniform["sweep"]],
        "uniform_reference": [r["uniform_respect"] for r in cmp_level["sweep"]],
        "canonical_density": payload["canonical_density"],
    }
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # 3) Operating range: vary N over ~2 orders of magnitude of edge count.
    n_list = [4, 8, 16, 32, 64]
    scale_densities = [0.1, 0.2, 0.3, 0.5]
    scale = []
    for n in n_list:
        for d in scale_densities:
            dags = []
            for k in range(16):
                rng = np.random.default_rng(1000 * n + int(100 * d) + k)
                dags.append(task._sample_dag(rng, n, d))
            tr = _score_on_dags(task, dags, level_attention_fn)
            scale.append({"n": n, "density": float(d), "topo_respect": float(tr)})
    with open(run_dir / "scale.json", "w") as f:
        json.dump(scale, f, indent=2)

    # 4) Canonical per-DAG artefacts for the heatmap viz.
    batch = task.generate(task.EVAL_SEED)
    ci = list(batch.densities).index(batch.canonical_density)
    can_dags = batch.dags[ci]
    adjs, levels, attns = [], [], []
    for adj in can_dags:
        n = adj.shape[0]
        adj_t = torch.as_tensor(adj, dtype=torch.float32, device=DEVICE)
        lvl = _compute_levels(adj_t, n).detach().cpu().numpy()
        adjs.append(adj.astype(np.float32))
        levels.append(lvl)
        attns.append(level_attention_fn(adj, n))
    np.savez(
        run_dir / "canonical.npz",
        adjacency=np.stack(adjs),
        levels=np.stack(levels),
        attention=np.stack(attns),
        canonical_density=float(batch.canonical_density),
    )

    print("topo_respect canonical:", payload["sweep"][ci]["topo_respect"])
    print("comparison level:", comparison["level_attention"])
    print("comparison direct:", comparison["direct_edge_strawman"])
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
