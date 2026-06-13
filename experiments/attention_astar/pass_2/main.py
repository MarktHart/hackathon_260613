"""attention_astar / pass_2 — hand-built A* attention circuit on the GPU.

Mechanism (no training): for each grid we compute, with pure torch tensor ops
on CUDA, the A* f-value f = g + h for every free cell, where
    g = true shortest-path distance from the agent (min-plus / BFS relaxation),
    h = Manhattan distance to the goal.
Attention = softmax(-beta * f - eps * g) over reachable free cells, with the
agent's own cell suppressed. Low-f cells (the cells A* expands) get the mass;
the -eps*g tie-breaker pushes the argmax onto the immediate optimal neighbour.
"""

import numpy as np
import torch

DEVICE = "cuda"


def make_model_fn(beta: float = 3.0, eps: float = 0.25,
                  use_g: bool = True, use_h: bool = True):
    """Return a ModelFn: grids [B,N,N,4] float32 -> attention [B,N,N] float32."""

    def model_fn(grids: np.ndarray) -> np.ndarray:
        x = torch.as_tensor(np.asarray(grids), dtype=torch.float32, device=DEVICE)
        B, N, _, _ = x.shape
        obst = x[..., 0] > 0.5            # [B,N,N]
        goal_ch = x[..., 2]
        agent_ch = x[..., 3]
        free = ~obst

        ar = torch.arange(N, device=DEVICE)
        rows = ar.view(1, N, 1).float()
        cols = ar.view(1, 1, N).float()

        gflat = goal_ch.view(B, -1).argmax(dim=1)
        gr = (gflat // N).float().view(B, 1, 1)
        gc = (gflat % N).float().view(B, 1, 1)
        h = (rows - gr).abs() + (cols - gc).abs()        # [B,N,N] Manhattan to goal

        # --- g: shortest free-path distance from the agent via min-plus relax ---
        INF = 1e9
        aflat = agent_ch.view(B, -1).argmax(dim=1)
        ai = aflat // N
        aj = aflat % N
        dist = torch.full((B, N, N), INF, device=DEVICE)
        bidx = torch.arange(B, device=DEVICE)
        dist[bidx, ai, aj] = 0.0
        for _ in range(N * N):            # 64 iters >= max graph diameter
            up = torch.full_like(dist, INF);   up[:, 1:, :] = dist[:, :-1, :]
            dn = torch.full_like(dist, INF);   dn[:, :-1, :] = dist[:, 1:, :]
            lf = torch.full_like(dist, INF);   lf[:, :, 1:] = dist[:, :, :-1]
            rt = torch.full_like(dist, INF);   rt[:, :, :-1] = dist[:, :, 1:]
            neigh = torch.minimum(torch.minimum(up, dn), torch.minimum(lf, rt)) + 1.0
            dist = torch.minimum(dist, neigh)
            dist = torch.where(obst, torch.full_like(dist, INF), dist)
            dist[bidx, ai, aj] = 0.0

        g = dist
        reachable = free & (dist < INF * 0.5) & (~(agent_ch > 0.5))

        g_term = g if use_g else torch.zeros_like(g)
        h_term = h if use_h else torch.zeros_like(h)
        f = g_term + h_term

        logit = -beta * f - eps * g_term
        neg_inf = torch.full_like(logit, -float("inf"))
        logit = torch.where(reachable, logit, neg_inf)
        m = logit.view(B, -1).max(dim=1).values.view(B, 1, 1)
        m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        attn = torch.exp(logit - m)
        attn = torch.where(reachable, attn, torch.zeros_like(attn))
        return attn.detach().cpu().numpy().astype(np.float32)

    return model_fn


if __name__ == "__main__":
    import json
    from agentic.experiments import load_task, record_benchmark, results_dir

    task = load_task(__file__)
    run_dir = results_dir(__file__)

    full_fn = make_model_fn()
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)

    # --- Ablations for the baseline-comparison story (canonical density 0.2) ---
    def canon(p):
        rec = p["sweep"][p["canonical_density_index"]]
        return {
            "alignment": rec["heuristic_alignment"],
            "entropy": rec["attention_entropy"],
            "top1": rec["top1_optimal_rate"],
            "top3": rec["top3_optimal_rate"],
            "path_gap": rec["path_optimality_gap"],
            "baseline": rec["linear_baseline_alignment"],
        }

    ablation = {
        "full (g+h)": canon(payload),
        "h only": canon(task.evaluate(make_model_fn(use_g=False))),
        "g only": canon(task.evaluate(make_model_fn(use_h=False))),
        "uniform": canon(task.evaluate(task.random_model_fn())),
    }
    with open(run_dir / "ablation_summary.json", "w") as f:
        json.dump(ablation, f, indent=2)

    # --- One example grid for the demo heatmaps ---
    batch = task.generate(seed=task.EVAL_SEED)
    idx = int(np.where(np.isclose(batch.obstacle_densities, 0.2))[0][0])
    grid = batch.grids[idx]
    attn = full_fn(grid[None])[0]
    agent = tuple(int(v) for v in batch.agent_positions[idx])
    goal = tuple(int(v) for v in batch.goal_positions[idx])
    obstacle_mask = grid[:, :, 0] > 0.5
    f_vals = task._f_values(obstacle_mask, agent, goal)
    f_save = np.where(np.isfinite(f_vals), f_vals, -1.0)
    np.savez(run_dir / "demo_example.npz",
             obstacle=obstacle_mask.astype(np.float32),
             attn=attn.astype(np.float32),
             f_values=f_save.astype(np.float32),
             agent=np.array(agent), goal=np.array(goal),
             start=np.array([int(v) for v in batch.start_positions[idx]]),
             optimal_next=np.array(list(batch.optimal_next_cells[idx])))

    print("done", json.dumps({k: round(v["alignment"], 3) for k, v in ablation.items()}))
