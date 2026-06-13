"""attention_knapsack / pass_2

Hand-built attention-style selection circuit for the 0/1 knapsack.

Mechanism (no training; weights are hand-set):
  1. Greedy-ratio initialisation, computed as a sequential selection over
     items sorted by value/weight (the LP-relaxation order).
  2. `T` rounds of *attention-guided 1-exchange local search*. At each round
     every instance forms a query (its current solution + remaining capacity)
     and attends over a grid of candidate moves -- "add item j" and
     "swap out item r for item j". Each move gets a score = value gain, masked
     to -inf when it violates capacity. A hard-attention argmax (temperature -> 0
     softmax) picks the single best improving move and applies it.

Because only strictly value-improving, capacity-feasible moves are ever
applied, the selection is feasible by construction and its value is >= the
greedy baseline on every instance -- it strictly beats greedy whenever any
improving 1-exchange exists, which is what lifts it above the benchmark floor.

All real compute runs in torch on CUDA.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; never fall back to CPU


# ──────────────────────────────────────────────────────────────────────
# The hand-built circuit
# ──────────────────────────────────────────────────────────────────────

def _greedy_ratio_init(w: torch.Tensor, v: torch.Tensor, cap: float):
    """Sequential greedy over the value/weight order. Returns (sel, used)."""
    B, N = w.shape
    ar = torch.arange(B, device=w.device)
    ratios = v / (w + 1e-8)
    order = torch.argsort(ratios, dim=1, descending=True)        # (B, N)
    w_ord = torch.gather(w, 1, order)                            # (B, N)

    sel = torch.zeros_like(w)
    used = torch.zeros(B, device=w.device)
    for step in range(N):
        idx = order[:, step]                                    # (B,)
        wi = w_ord[:, step]                                     # (B,)
        fits = (used + wi) <= cap + 1e-6
        sel[ar, idx] = fits.float()
        used = used + torch.where(fits, wi, torch.zeros_like(wi))
    return sel, used


def _attention_refine(sel: torch.Tensor, w: torch.Tensor, v: torch.Tensor,
                      cap: float, rounds: int = 64):
    """Attention-guided 1-exchange local search (hard attention over moves)."""
    B, N = w.shape
    ar = torch.arange(B, device=w.device)

    # Augment the "remove" axis with a virtual slot N == "remove nothing",
    # so a pure add and a swap share one move grid of shape (B, N+1, N).
    zcol = torch.zeros(B, 1, device=w.device)
    rem_w = torch.cat([w, zcol], dim=1)                          # (B, N+1)
    rem_v = torch.cat([v, zcol], dim=1)                          # (B, N+1)

    for _ in range(rounds):
        used = (sel * w).sum(1)                                  # (B,)
        sel_ext = torch.cat([sel, torch.ones(B, 1, device=w.device)], dim=1)
        add_valid = (sel == 0)                                   # (B, N): j unselected

        # value gain of removing r and adding j
        gain = v.unsqueeze(1) - rem_v.unsqueeze(2)              # (B, N+1, N)
        new_w = used.view(B, 1, 1) - rem_w.unsqueeze(2) + w.unsqueeze(1)
        feasible = new_w <= cap + 1e-6
        valid = (sel_ext.unsqueeze(2).bool()                    # remove slot valid
                 & add_valid.unsqueeze(1)                       # add slot unselected
                 & feasible)
        score = torch.where(valid, gain, torch.full_like(gain, -1e9))

        flat = score.view(B, -1)
        best, arg = flat.max(dim=1)                             # hard attention
        improve = best > 1e-6
        if not bool(improve.any()):
            break
        r_idx = torch.div(arg, N, rounding_mode="floor")
        j_idx = arg % N

        sel[ar[improve], j_idx[improve]] = 1.0                  # apply the add
        rem = improve & (r_idx < N)                             # apply the remove
        sel[ar[rem], r_idx[rem]] = 0.0
    return sel


def make_model_fn():
    def my_model_fn(weights, values, capacity):
        w = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
        v = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)
        cap = float(capacity)
        sel, _ = _greedy_ratio_init(w, v, cap)
        sel = _attention_refine(sel, w, v, cap, rounds=64)
        return sel.detach().cpu().numpy().astype(np.float32)
    return my_model_fn


# ──────────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────────

def _save_example(task, model_fn, run_dir):
    """Save one canonical instance where refinement strictly beats greedy, so
    the Demo tab can show the mechanism doing something a heuristic can't."""
    batch = task.generate(seed=42, capacity_frac=task.CANONICAL_CAPACITY_FRAC)
    w, v, cap = batch.weights, batch.values, batch.capacity

    greedy_sel = task.greedy_baseline_fn(w, v, cap) >= 0.5
    refined_sel = model_fn(w, v, cap) >= 0.5
    opt_sel = batch.optimal_selections

    def val(sel_row, i):
        feas = float((sel_row * w[i]).sum()) <= cap + 1e-6
        return float((sel_row * v[i]).sum()) if feas else 0.0

    # pick an instance where refined strictly improves on greedy
    pick = 0
    for i in range(w.shape[0]):
        if val(refined_sel[i], i) > val(greedy_sel[i], i) + 1e-6:
            pick = i
            break

    example = {
        "capacity": float(cap),
        "weights": w[pick].astype(float).tolist(),
        "values": v[pick].astype(float).tolist(),
        "greedy_sel": greedy_sel[pick].astype(int).tolist(),
        "refined_sel": refined_sel[pick].astype(int).tolist(),
        "optimal_sel": opt_sel[pick].astype(int).tolist(),
        "greedy_value": val(greedy_sel[pick], pick),
        "refined_value": val(refined_sel[pick], pick),
        "optimal_value": float(batch.optimal_values[pick]),
    }
    (run_dir / "example.json").write_text(json.dumps(example, indent=2))


def main() -> None:
    task = load_task(__file__)
    model_fn = make_model_fn()

    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    _save_example(task, model_fn, run_dir)
    record_benchmark(__file__, run_dir, payload)
    print(f"Wrote benchmark + example artefacts to {run_dir}")


if __name__ == "__main__":
    main()
