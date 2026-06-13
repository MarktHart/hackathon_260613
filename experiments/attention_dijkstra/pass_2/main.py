"""attention_dijkstra / pass_2 — soft-min ATTENTION relaxation.

Mechanism (delta from experiments/base_model.py):
    base_model's self-attention computes  softmax(Q Kᵀ) · V .
    Here we replace the three pieces with their min-plus / shortest-path analogue,
    keeping the *softmax-attention* shape intact:

      * logits[u, v] = -beta * (d_u + w_{u->v})            (QKᵀ  ->  negated relax cost)
      * attn        = softmax(logits, over predecessors u)  (the same softmax)
      * read-out    = soft-min = -1/beta * logsumexp(-beta*(d_u + w_{u->v}))
                                                            (V-aggregation -> min-plus)

    The block is weight-tied and applied recurrently for n-1 steps (the
    Bellman-Ford bound on the number of relaxations any shortest path needs),
    and the MLP is dropped.  `beta` is the attention *temperature*: beta -> inf
    recovers hard Dijkstra/Bellman-Ford, finite beta is genuine softmax
    attention.  This is the soft-min the goal frames, NOT a hard torch.min.

Everything runs in torch float64 on CUDA.
"""

import json
import os

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
DTYPE = torch.float64
BETA = 1000.0          # attention temperature used for the headline model_fn
LARGE = 1.0e9          # finite stand-in for "no edge" / "not yet reached"


# --------------------------------------------------------------------------- #
# Core soft-min attention relaxation (one weight-tied attention block, looped) #
# --------------------------------------------------------------------------- #
def _to_tensor(weights: np.ndarray) -> torch.Tensor:
    W = torch.as_tensor(np.asarray(weights, dtype=np.float64), dtype=DTYPE, device=DEVICE)
    return torch.where(torch.isinf(W), torch.full_like(W, LARGE), W)


def softmin_relax(W: torch.Tensor, source: int, hops: int, beta: float,
                  record_history: bool = False):
    """Iterated soft-min attention.  Returns dist (and optional per-hop history)."""
    n = W.shape[0]
    dist = torch.full((n,), LARGE, dtype=DTYPE, device=DEVICE)
    dist[source] = 0.0
    history = [dist.clone()] if record_history else None
    for _ in range(hops):
        # cost[u, v] = best-known distance to u  +  edge cost u -> v
        cost = dist.unsqueeze(1) + W                       # (n_pred u, n_target v)
        # soft-argmin attention over predecessors, soft-min read-out:
        dist = -(1.0 / beta) * torch.logsumexp(-beta * cost, dim=0)   # (n_target,)
        if record_history:
            history.append(dist.clone())
    if record_history:
        return dist, history
    return dist


def final_attention(W: torch.Tensor, dist: torch.Tensor, beta: float) -> torch.Tensor:
    """Attention matrix at the converged state: attn[u, v] = how much v attends to u."""
    cost = dist.unsqueeze(1) + W
    return torch.softmax(-beta * cost, dim=0)


def make_model_fn(beta: float = BETA):
    """The attempt's contribution: weights, source -> predicted distances."""
    def model_fn(weights: np.ndarray, source: int) -> np.ndarray:
        n = int(np.asarray(weights).shape[0])
        W = _to_tensor(weights)
        hops = max(1, n - 1)                       # adaptive: Bellman-Ford bound
        dist = softmin_relax(W, int(source), hops, beta)
        return dist.detach().cpu().numpy()
    return model_fn


# --------------------------------------------------------------------------- #
# Extra analyses for the Demo (faithfulness / operating-range / temperature)   #
# --------------------------------------------------------------------------- #
def _group_by_n(task):
    """Return {n: [(weights, source, true, mask), ...]} for the canonical batch."""
    batch = task.generate(seed=task.EVAL_SEED)
    groups = {n: [] for n in task.N_NODES_SWEEP}
    for weights, source, n in zip(batch.weights, batch.sources, batch.n_nodes):
        true = task._shortest_paths(weights, source)
        mask = np.isfinite(true)
        mask[source] = False
        groups[n].append((weights, int(source), true, mask))
    return groups


def hop_ablation(task, groups, beta: float) -> dict:
    """Distance accuracy as a function of #relaxation hops, per graph size.

    This is the causal/faithfulness check: knock out hops (shrink propagation
    depth) and watch accuracy collapse toward the no-propagation one-hop level.
    """
    curves, hops_axis = {}, {}
    for n in task.N_NODES_SWEEP:
        max_hops = max(1, n - 1)
        accs = np.zeros(max_hops + 1, dtype=np.float64)   # hop 0 .. max_hops
        for weights, source, true, mask in groups[n]:
            W = _to_tensor(weights)
            _, hist = softmin_relax(W, source, max_hops, beta, record_history=True)
            for h, d in enumerate(hist):
                pred = d.detach().cpu().numpy()
                accs[h] += task._accuracy(pred, true, mask)
        accs /= len(groups[n])
        curves[str(n)] = accs.tolist()
        hops_axis[str(n)] = list(range(max_hops + 1))
    return {"curves": curves, "hops": hops_axis}


def beta_sweep(task, groups, betas) -> dict:
    """Accuracy/ordering vs attention temperature at the canonical slice (n=16)."""
    cn = task.CANONICAL_N
    acc_out, corr_out = [], []
    for beta in betas:
        accs, corrs = [], []
        for weights, source, true, mask in groups[cn]:
            n = weights.shape[0]
            W = _to_tensor(weights)
            pred = softmin_relax(W, source, max(1, n - 1), float(beta)).detach().cpu().numpy()
            accs.append(task._accuracy(pred, true, mask))
            corrs.append(task._spearman(pred[mask], true[mask]) if np.any(mask) else 0.0)
        acc_out.append(float(np.mean(accs)))
        corr_out.append(float(np.mean(corrs)))
    return {"betas": [float(b) for b in betas], "accuracy": acc_out, "order_corr": corr_out}


def example_graph(task, groups, beta: float) -> dict:
    """One canonical n=16 graph: true vs pred vs one-hop, per-hop history, attention."""
    cn = task.CANONICAL_N
    weights, source, true, mask = groups[cn][0]
    n = weights.shape[0]
    W = _to_tensor(weights)
    dist, hist = softmin_relax(W, source, max(1, n - 1), beta, record_history=True)
    attn = final_attention(W, dist, beta).detach().cpu().numpy()
    pred = dist.detach().cpu().numpy()
    onehop = task._onehop_baseline(weights, source)
    history = np.stack([h.detach().cpu().numpy() for h in hist], axis=0)  # (hops+1, n)
    predecessor = attn.argmax(axis=0)                                     # v -> best pred u
    adj = np.where(np.isfinite(weights), weights, np.nan)
    return {
        "n": int(n),
        "source": int(source),
        "true": true,
        "pred": pred,
        "onehop": onehop,
        "mask": mask,
        "history": history,
        "attn": attn,
        "predecessor": predecessor,
        "adj": adj,
    }


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)
    run_dir_s = str(run_dir)

    # 1) Headline benchmark via the goal's evaluator.
    model_fn = make_model_fn(BETA)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # 2) Extra analyses for the Demo.
    groups = _group_by_n(task)
    abl = hop_ablation(task, groups, BETA)
    cn = task.CANONICAL_N
    base_accs = [task._accuracy(task._onehop_baseline(w, s), t, m) for (w, s, t, m) in groups[cn]]
    abl["onehop_baseline_canonical"] = float(np.mean(base_accs))
    abl["beta"] = BETA
    with open(os.path.join(run_dir_s, "hop_ablation.json"), "w") as f:
        json.dump(abl, f)

    bsw = beta_sweep(task, groups, [1, 2, 5, 10, 20, 50, 100, 300, 1000])
    with open(os.path.join(run_dir_s, "beta_sweep.json"), "w") as f:
        json.dump(bsw, f)

    ex = example_graph(task, groups, BETA)
    np.savez(
        os.path.join(run_dir_s, "example_graph.npz"),
        n=ex["n"], source=ex["source"], true=ex["true"], pred=ex["pred"],
        onehop=ex["onehop"], mask=ex["mask"], history=ex["history"],
        attn=ex["attn"], predecessor=ex["predecessor"], adj=ex["adj"], beta=BETA,
    )

    sweep = {r["n_nodes"]: r for r in payload["sweep"]}
    summary = {
        "beta": BETA,
        "distance_accuracy_canonical": sweep[cn]["distance_accuracy"],
        "order_correlation_canonical": sweep[cn]["order_correlation"],
        "onehop_baseline_canonical": abl["onehop_baseline_canonical"],
        "distance_accuracy_per_n": {str(r["n_nodes"]): r["distance_accuracy"]
                                    for r in payload["sweep"]},
    }
    with open(os.path.join(run_dir_s, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("pass_2 done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
