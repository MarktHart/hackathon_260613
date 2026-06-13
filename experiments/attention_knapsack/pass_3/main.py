"""attention_knapsack / pass_3

A hand-built *attention swap-circuit* that solves 0/1 knapsack by iterative
local refinement, expressed entirely as masked attention operations on the GPU.

The mechanism
-------------
1. **Greedy ratio initialisation** (the no-mechanism baseline) gives a feasible
   starting selection S.
2. **Attention swap head.** Treat the currently-selected items as *queries* and
   the unselected items as *keys*. The attention score for the (i in S, j not in
   S) pair is the value delta `v_j - v_i`, masked to -inf unless the swap keeps
   the knapsack feasible (`W - w_i + w_j <= cap`) and strictly improves value.
   A pure *add* (an empty query slot) is the special case `i = none`.
3. Apply the move, recompute residual capacity, repeat until no instance has an
   improving move (or a step cap is hit).

Because we start from the greedy baseline and only ever apply value-increasing,
feasibility-preserving moves, the circuit is provably >= greedy pointwise and
keeps feasible_rate = 1.0 — exactly the two things the previous pass failed.
The swap head uses only top-1 attention and feasibility masking — no learned
weights, no MLP, no residual stream — tying the optimality gain directly to a
single, interpretable attention operation.

Everything runs in torch on CUDA. NumPy only at the task boundary.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback
NEG = -1e30


# ──────────────────────────────────────────────────────────────────────
# Hand-built attention swap-circuit (all GPU torch)
# ──────────────────────────────────────────────────────────────────────

def _greedy_init(weights: torch.Tensor, values: torch.Tensor, cap: float) -> torch.Tensor:
    """Vectorised greedy value/weight-ratio policy → feasible 0/1 selection."""
    B, n = weights.shape
    ratios = values / (weights + 1e-8)
    order = torch.argsort(-ratios, dim=1)                 # (B, n) best ratio first
    w_sorted = torch.gather(weights, 1, order)
    sel_sorted = torch.zeros_like(weights)
    w_accum = torch.zeros(B, device=weights.device)
    for k in range(n):                                    # n=16 sequential steps
        wk = w_sorted[:, k]
        fits = (w_accum + wk) <= cap + 1e-6
        sel_sorted[:, k] = fits.float()
        w_accum = w_accum + fits.float() * wk
    # scatter the sorted selection back into the original order
    sel = torch.zeros_like(weights)
    for k in range(n):
        sel.scatter_(1, order[:, k].unsqueeze(1), sel_sorted[:, k].unsqueeze(1))
    return sel


def attention_swap_solver(
    weights: torch.Tensor,
    values: torch.Tensor,
    cap: float,
    max_steps: int = 64,
) -> torch.Tensor:
    """Local-search refinement via a masked top-1 attention head. Returns 0/1."""
    B, n = weights.shape
    sel = _greedy_init(weights, values, cap)
    b_idx = torch.arange(B, device=weights.device)

    for _ in range(max_steps):
        W = (sel * weights).sum(dim=1)                    # (B,) current weight

        # ---- swap head: queries = selected i, keys = unselected j ----
        vi = values.unsqueeze(2)                          # (B, n, 1)  query value
        vj = values.unsqueeze(1)                          # (B, 1, n)  key value
        wi = weights.unsqueeze(2)
        wj = weights.unsqueeze(1)
        delta = vj - vi                                   # (B, n, n)  attention score
        new_w = W[:, None, None] - wi + wj
        sel_i = sel.unsqueeze(2) > 0.5                    # i is selected (query valid)
        unsel_j = sel.unsqueeze(1) < 0.5                  # j is unselected (key valid)
        feasible = new_w <= cap + 1e-6
        valid = sel_i & unsel_j & feasible & (delta > 1e-9)
        swap_score = torch.where(valid, delta, torch.full_like(delta, NEG))
        flat = swap_score.view(B, -1)
        best_swap_val, best_swap_idx = flat.max(dim=1)
        i_idx = torch.div(best_swap_idx, n, rounding_mode="floor")
        j_idx = best_swap_idx % n

        # ---- add head: empty query, just insert an unselected item ----
        can_add = (sel < 0.5) & ((W[:, None] + weights) <= cap + 1e-6)
        add_score = torch.where(can_add, values, torch.full_like(values, NEG))
        best_add_val, best_add_j = add_score.max(dim=1)

        # ---- top-1 attention: pick the single best improving move ----
        take_swap = best_swap_val >= best_add_val
        move_val = torch.where(take_swap, best_swap_val, best_add_val)
        improving = move_val > 1e-9
        if not bool(improving.any()):
            break

        swap_mask = improving & take_swap
        add_mask = improving & (~take_swap)
        if bool(swap_mask.any()):
            rows = b_idx[swap_mask]
            sel[rows, i_idx[swap_mask]] = 0.0
            sel[rows, j_idx[swap_mask]] = 1.0
        if bool(add_mask.any()):
            rows = b_idx[add_mask]
            sel[rows, best_add_j[add_mask]] = 1.0

    return sel


# ──────────────────────────────────────────────────────────────────────
# model_fn matching the task contract: (weights, values, capacity) -> (B,n)
# ──────────────────────────────────────────────────────────────────────

def make_model_fn():
    def model_fn(weights, values, capacity):
        w = torch.as_tensor(weights, dtype=torch.float32, device=DEVICE)
        v = torch.as_tensor(values, dtype=torch.float32, device=DEVICE)
        sel = attention_swap_solver(w, v, float(capacity))
        return sel.detach().cpu().numpy().astype(np.float32)

    return model_fn


def _opt(rec):
    return max(0.0, min(1.0, 1.0 - float(rec["optimality_gap"])))


def main() -> None:
    task = load_task(__file__)
    model_fn = make_model_fn()

    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)

    # Self-contained artefact for the Demo tab (curves the viz needs).
    sweep = payload["sweep"]
    base_sweep = payload["baseline_sweep"]
    demo_data = {
        "sweep_fracs": [r["capacity_frac"] for r in sweep],
        "model_optimality": [_opt(r) for r in sweep],
        "baseline_optimality": [_opt(r) for r in base_sweep],
        "model_feasible": [float(r["feasible_rate"]) for r in sweep],
        "model_value": [float(r["model_value"]) for r in sweep],
        "optimal_value": [float(r["optimal_value"]) for r in sweep],
        "baseline_value": [float(r["model_value"]) for r in base_sweep],
        "canonical_optimality": _opt(payload["canonical"]),
        "baseline_canonical_optimality": _opt(payload["baseline_canonical"]),
        "canonical_feasible": float(payload["canonical"]["feasible_rate"]),
        "robustness": float(np.mean([_opt(r) for r in sweep])),
    }
    (run_dir / "demo_data.json").write_text(json.dumps(demo_data, indent=2))

    record_benchmark(__file__, run_dir, payload)
    print("canonical optimality :", demo_data["canonical_optimality"])
    print("greedy   optimality  :", demo_data["baseline_canonical_optimality"])
    print("robustness (headline):", demo_data["robustness"])
    print("canonical feasible   :", demo_data["canonical_feasible"])
    print("wrote", run_dir / "benchmark.json")


if __name__ == "__main__":
    main()