"""pass_2 — a hand-built ATTENTION circuit that computes the transitive closure.

Unlike first_pass (which was raw (I+A)^5 matrix powers and avoided attention
entirely), this attempt expresses the closure as a *stack of self-attention
layers* — the smallest delta from `experiments/base_model.py`:

    base_model.Attention, but with:
      * Q/K replaced by an adjacency-derived ADDITIVE attention bias
        (a graph relative-position bias: score(i,j)=0 if j is i or a neighbour,
         -inf otherwise) -> softmax gives uniform attention over self+neighbours;
      * token embedding = identity one-hot (values carry node identity);
      * a hard-threshold activation after each attention write (support read-out);
      * the MLP dropped (a single attention op per layer is enough);
      * `depth` stacked layers, no causal mask, no RoPE.

One attention layer propagates information exactly one hop, so after L layers a
node's residual stream marks every node within L hops. For L >= diameter this is
the full connected component => the same-component affinity is exact. Crucially:

    * depth == 1  ->  the read-out is EXACTLY the 1-hop adjacency relation
                      (this IS the goal's strawman baseline), and
    * the circuit BREAKS the moment depth < diameter.

That depth knob is the built-in faithfulness/ablation handle: removing attention
hops causally collapses the closure back to adjacency. main.py records that
ablation (F1 vs depth, per diameter) alongside the contract payload, plus the
per-layer reachability maps for the heatmap viz.

Everything runs in torch on CUDA (no CPU fallback).
"""
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; never fall back to CPU
NEG_INF = -1.0e9
MAX_DEPTH = 8  # ablation/heatmap depth range (covers the diameter-5 slice w/ margin)


# --------------------------------------------------------------------------- #
# The attention circuit
# --------------------------------------------------------------------------- #
def _attention_pattern(adjacency: np.ndarray):
    """Build the (fixed) self-attention weights + identity values on GPU.

    score(i, j) = 0      if j == i or A[i, j] == 1   (self or 1-hop neighbour)
                = -inf   otherwise
    attn = softmax(score) -> uniform attention over the self+neighbour set.

    Returns (attn, values, n) with attn, values on CUDA.
    """
    adj = torch.as_tensor(adjacency, dtype=torch.float32, device=DEVICE)
    n = adj.shape[0]
    eye = torch.eye(n, dtype=torch.float32, device=DEVICE)
    allowed = (eye + adj) > 0  # self-loop + neighbours
    scores = torch.where(
        allowed, torch.zeros_like(adj), torch.full_like(adj, NEG_INF)
    )
    attn = torch.softmax(scores, dim=-1)  # row-uniform over allowed set
    values = eye  # identity token embeddings: value of node j is e_j
    return attn, values, n


def _run_layers(attn: torch.Tensor, values: torch.Tensor, depth: int) -> torch.Tensor:
    """Stack `depth` attention layers with a support (>0) read-out per layer.

    X_0 = identity. Each layer: X <- 1[ attn @ X > 0 ]. After `depth` layers,
    X[i, j] == 1 iff node j is within `depth` hops of node i.
    """
    x = values  # (n, n), identity
    for _ in range(depth):
        x = (attn @ x > 0).to(torch.float32)
    return x


def make_model_fn(depth=None):
    """Return a contract-shaped model_fn at a fixed attention depth.

    depth=None -> use n layers, guaranteeing depth >= diameter for ANY graph
    size (max possible diameter is n-1), so the closure is exact regardless of
    N / diameter. This is what the benchmark payload uses.
    """

    def model_fn(adjacency: np.ndarray) -> np.ndarray:
        attn, values, n = _attention_pattern(adjacency)
        d = n if depth is None else depth
        x = _run_layers(attn, values, d)
        return x.detach().cpu().numpy()  # (n, n) 0/1 same-component affinity

    return model_fn


def _reach_stack(adjacency: np.ndarray, max_depth: int) -> np.ndarray:
    """Per-layer reachability cube (max_depth, n, n) for the heatmap viz."""
    attn, values, n = _attention_pattern(adjacency)
    frames = []
    x = values
    for _ in range(max_depth):
        x = (attn @ x > 0).to(torch.float32)
        frames.append(x.detach().cpu().numpy().copy())
    return np.stack(frames, axis=0)


# --------------------------------------------------------------------------- #
# Scoring helper (mirrors benchmark._f1, kept local for the ablation artefact)
# --------------------------------------------------------------------------- #
def _f1(counts: dict) -> float:
    tp, fp, fn = float(counts["tp"]), float(counts["fp"]), float(counts["fn"])
    denom = 2.0 * tp + fp + fn
    return (2.0 * tp) / denom if denom > 0 else 0.0


def main() -> None:
    task = load_task(__file__)

    # 1) Contract payload: full-depth attention circuit (exact closure).
    payload = task.evaluate(make_model_fn(depth=None))

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    diameters = [rec["diameter"] for rec in payload["sweep"]]

    # 2) Faithfulness/ablation artefact: F1 vs attention depth, per diameter.
    #    depth=1 must coincide with the adjacency baseline; the curve must jump
    #    to 1.0 exactly at depth == diameter.
    depths = list(range(1, MAX_DEPTH + 1))
    per_depth = {d: task.evaluate(make_model_fn(depth=d)) for d in depths}

    model_f1 = {str(D): [] for D in diameters}
    baseline_f1 = {}
    for rec in payload["sweep"]:
        baseline_f1[str(rec["diameter"])] = _f1(rec["baseline"])
    for d in depths:
        for rec in per_depth[d]["sweep"]:
            model_f1[str(rec["diameter"])].append(_f1(rec["model"]))

    full_depth_f1 = {
        str(D): model_f1[str(D)][-1] for D in diameters
    }  # depth=MAX_DEPTH >= every diameter in the sweep
    headline = float(np.mean(list(full_depth_f1.values())))

    ablation = {
        "version": 1,
        "depths": depths,
        "diameters": diameters,
        "canonical_diameter": payload["canonical_diameter"],
        "model_f1": model_f1,          # {diam: [f1 at each depth]}
        "baseline_f1": baseline_f1,    # {diam: scalar} (== model_f1 at depth 1)
        "full_depth_f1": full_depth_f1,
        "transitive_closure_robustness": headline,
    }
    with open(run_dir / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # 3) Heatmap artefact: one representative graph per diameter, per-layer
    #    reachability + ground truth, reordered later by component in the app.
    batch = task.generate(seed=0)
    for diameter, graphs in batch.slices:
        adjacency, labels = graphs[0]
        stack = _reach_stack(adjacency, MAX_DEPTH)  # (MAX_DEPTH, n, n)
        truth = (labels[:, None] == labels[None, :]).astype(np.float32)
        np.savez(
            run_dir / f"reach_diam_{int(diameter)}.npz",
            reach=stack.astype(np.uint8),
            truth=truth.astype(np.uint8),
            adjacency=np.asarray(adjacency, dtype=np.uint8),
            labels=np.asarray(labels, dtype=np.int32),
            depths=np.asarray(depths, dtype=np.int32),
        )

    print(f"Done. transitive_closure_robustness={headline:.4f}. Results in {run_dir}")


if __name__ == "__main__":
    main()
