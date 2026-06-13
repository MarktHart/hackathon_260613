"""Synthetic A* navigation task for the attention_astar goal.

Exports:
    generate(seed) -> Batch          deterministic per seed
    evaluate(model_fn) -> dict        payload consumed by benchmark.score
    random_model_fn() -> ModelFn      uniform-attention reference model (no args)

Pure NumPy + stdlib. No torch, no GPU, no I/O.
"""

from __future__ import annotations

import math
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np

# A model_fn maps a [B, N, N, 4] grid batch to a [B, N, N] attention batch.
ModelFn = Callable[[np.ndarray], np.ndarray]

# ---- Canonical measurement condition (see README.md) ------------------------
N = 8
NUM_GRIDS = 128
DENSITIES = [0.0, 0.1, 0.2, 0.3, 0.4]
HEURISTIC = "manhattan"
CANONICAL_DENSITY = 0.2
EVAL_SEED = 42
MAX_GREEDY_STEPS = 64


@dataclass(frozen=True)
class Batch:
    grids: np.ndarray                                # [B, N, N, 4] float32
    obstacle_densities: np.ndarray                   # [B] float32
    start_positions: np.ndarray                      # [B, 2] int32
    goal_positions: np.ndarray                       # [B, 2] int32
    agent_positions: np.ndarray                      # [B, 2] int32
    optimal_next_cells: List[List[Tuple[int, int]]]  # len B
    astar_path_lengths: np.ndarray                   # [B] int32


# -----------------------------------------------------------------------------
# A* / search utilities (pure Python, deterministic)
# -----------------------------------------------------------------------------
def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


_NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _astar(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
    """Return (path, optimal_next_neighbours).

    path includes both endpoints; [] if unreachable. optimal_next_neighbours
    are the free neighbours of `start` lying on at least one optimal path.
    """
    n = grid.shape[0]
    open_set = [(_manhattan(start, goal), 0, start)]
    came_from = {start: None}
    g_score = {start: 0}

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current == goal:
            break
        if g > g_score.get(current, math.inf):
            continue
        r, c = current
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not grid[nr, nc]:
                neighbour = (nr, nc)
                tentative = g + 1
                if tentative < g_score.get(neighbour, math.inf):
                    g_score[neighbour] = tentative
                    came_from[neighbour] = current
                    heapq.heappush(
                        open_set,
                        (tentative + _manhattan(neighbour, goal), tentative, neighbour),
                    )

    if goal not in came_from:
        return [], []

    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()

    best = g_score[goal]

    # Distance from every reachable cell to the goal (uniform-cost BFS *from*
    # the goal). Needed to decide which neighbours of `start` lie on an optimal
    # path: a neighbour's distance-from-agent is trivially 1, which says nothing
    # about whether stepping there keeps the agent on a shortest route.
    dist_to_goal = {goal: 0}
    dq = deque([goal])
    while dq:
        cr, cc = dq.popleft()
        for dr, dc in _NEIGHBOURS:
            nr, nc = cr + dr, cc + dc
            if (0 <= nr < n and 0 <= nc < n and not grid[nr, nc]
                    and (nr, nc) not in dist_to_goal):
                dist_to_goal[(nr, nc)] = dist_to_goal[(cr, cc)] + 1
                dq.append((nr, nc))

    sr, sc = start
    optimal_next = []
    for dr, dc in _NEIGHBOURS:
        nr, nc = sr + dr, sc + dc
        if 0 <= nr < n and 0 <= nc < n and not grid[nr, nc]:
            dn = dist_to_goal.get((nr, nc))
            # On an optimal path iff stepping there strictly closes the gap:
            # 1 + dist(neighbour -> goal) == dist(start -> goal).
            if dn is not None and 1 + dn == best:
                optimal_next.append((nr, nc))
    return path, optimal_next


def _f_values(grid: np.ndarray, agent: Tuple[int, int], goal: Tuple[int, int]) -> np.ndarray:
    """f = g + h for every free cell; inf for obstacles / unreachable."""
    n = grid.shape[0]
    g = np.full((n, n), math.inf, dtype=float)
    g[agent] = 0.0
    pq = [(0, agent)]
    while pq:
        d, (r, c) = heapq.heappop(pq)
        if d > g[r, c]:
            continue
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not grid[nr, nc]:
                nd = d + 1
                if nd < g[nr, nc]:
                    g[nr, nc] = nd
                    heapq.heappush(pq, (nd, (nr, nc)))

    f = np.full((n, n), math.inf, dtype=float)
    for r in range(n):
        for c in range(n):
            if not grid[r, c] and math.isfinite(g[r, c]):
                f[r, c] = g[r, c] + _manhattan((r, c), goal)
    return f


# -----------------------------------------------------------------------------
# Data generation
# -----------------------------------------------------------------------------
def generate(seed: int = 0) -> Batch:
    """Deterministic batch for a given seed.

    Produces ~NUM_GRIDS grids split evenly across DENSITIES at the canonical
    grid size. Grids whose agent cannot reach the goal are resampled.
    """
    rng = np.random.default_rng(seed)
    per_density = NUM_GRIDS // len(DENSITIES)

    grids_list, dens_list = [], []
    starts, goals, agents = [], [], []
    optimal_next_list, astar_lens = [], []

    for density in DENSITIES:
        made = 0
        attempts = 0
        while made < per_density and attempts < per_density * 200:
            attempts += 1
            obstacle_mask = rng.random((N, N)) < density
            free = [(r, c) for r in range(N) for c in range(N) if not obstacle_mask[r, c]]
            if len(free) < 3:
                continue
            i, j, k = rng.choice(len(free), size=3, replace=False)
            start = free[int(i)]
            goal = free[int(j)]
            agent = free[int(k)]

            path, optimal_next = _astar(obstacle_mask, agent, goal)
            if not path or not optimal_next:
                continue

            grid = np.zeros((N, N, 4), dtype=np.float32)
            grid[:, :, 0] = obstacle_mask.astype(np.float32)
            grid[start[0], start[1], 1] = 1.0
            grid[goal[0], goal[1], 2] = 1.0
            grid[agent[0], agent[1], 3] = 1.0

            grids_list.append(grid)
            dens_list.append(density)
            starts.append(start)
            goals.append(goal)
            agents.append(agent)
            optimal_next_list.append(optimal_next)
            astar_lens.append(len(path) - 1)
            made += 1

    return Batch(
        grids=np.stack(grids_list),
        obstacle_densities=np.array(dens_list, dtype=np.float32),
        start_positions=np.array(starts, dtype=np.int32),
        goal_positions=np.array(goals, dtype=np.int32),
        agent_positions=np.array(agents, dtype=np.int32),
        optimal_next_cells=optimal_next_list,
        astar_path_lengths=np.array(astar_lens, dtype=np.int32),
    )


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 0.0
    # Constant input has no rank structure; argsort would invent order-based
    # ranks and report a spurious correlation. Treat as uncorrelated.
    if x.max() == x.min() or y.max() == y.min():
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    sx, sy = rx.std(), ry.std()
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return float(np.mean((rx - rx.mean()) * (ry - ry.mean())) / (sx * sy))


def _alignment(attn: np.ndarray, f_vals: np.ndarray, free: np.ndarray) -> float:
    if not np.any(free):
        return 0.0
    return _spearman(attn[free], -f_vals[free])


def _greedy_path(attn: np.ndarray, obstacle_mask: np.ndarray,
                 agent: Tuple[int, int], goal: Tuple[int, int]) -> int:
    """Follow top-1 attention, masking obstacles and visited cells. Return steps,
    or -1 if the goal is not reached."""
    n = attn.shape[0]
    current = agent
    visited = {current}
    for step in range(1, MAX_GREEDY_STEPS + 1):
        masked = attn.copy()
        masked[obstacle_mask] = -np.inf
        for (vr, vc) in visited:
            masked[vr, vc] = -np.inf
        flat = int(np.argmax(masked))
        nr, nc = divmod(flat, n)
        if not np.isfinite(masked[nr, nc]):
            return -1
        current = (nr, nc)
        visited.add(current)
        if current == goal:
            return step
    return -1


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate(model_fn: ModelFn) -> dict:
    """Run model_fn over the canonical batch; return the benchmark payload."""
    batch = generate(seed=EVAL_SEED)
    B = batch.grids.shape[0]

    raw = np.asarray(model_fn(batch.grids), dtype=np.float64)
    if raw.shape != (B, N, N):
        raise ValueError(f"model_fn returned shape {raw.shape}, expected ({B}, {N}, {N})")
    if not np.all(np.isfinite(raw)):
        raise ValueError("model_fn returned non-finite attention values")
    raw = np.clip(raw, 0.0, None)
    sums = raw.sum(axis=(1, 2), keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    attn_batch = raw / sums

    sweep = []
    for density in DENSITIES:
        idx = np.where(np.isclose(batch.obstacle_densities, density))[0]
        if idx.size == 0:
            continue

        entropies, aligns, base_aligns = [], [], []
        top1, top3, gaps = [], [], []

        for i in idx:
            attn = attn_batch[i]
            obstacle_mask = batch.grids[i, :, :, 0] > 0.5
            agent = (int(batch.agent_positions[i][0]), int(batch.agent_positions[i][1]))
            goal = (int(batch.goal_positions[i][0]), int(batch.goal_positions[i][1]))
            opt_next = set(batch.optimal_next_cells[i])
            astar_len = int(batch.astar_path_lengths[i])

            entropies.append(_entropy(attn.flatten()))

            f_vals = _f_values(obstacle_mask, agent, goal)
            free = (~obstacle_mask) & np.isfinite(f_vals)
            aligns.append(_alignment(attn, f_vals, free))

            uniform = free.astype(np.float64)
            base_aligns.append(_alignment(uniform, f_vals, free))

            peak = divmod(int(np.argmax(attn)), N)
            top1.append(1.0 if peak in opt_next else 0.0)
            top3_idx = np.argsort(attn.flatten())[-3:]
            top3_cells = {divmod(int(t), N) for t in top3_idx}
            top3.append(1.0 if (top3_cells & opt_next) else 0.0)

            steps = _greedy_path(attn, obstacle_mask, agent, goal)
            if steps > 0 and astar_len > 0:
                gaps.append((steps - astar_len) / astar_len)
            else:
                gaps.append(1.0)

        sweep.append({
            "obstacle_density": float(density),
            "n_grids": int(idx.size),
            "attention_entropy": float(np.mean(entropies)),
            "heuristic_alignment": float(np.mean(aligns)),
            "top1_optimal_rate": float(np.mean(top1)),
            "top3_optimal_rate": float(np.mean(top3)),
            "path_optimality_gap": float(np.mean(gaps)),
            "linear_baseline_alignment": float(np.mean(base_aligns)),
        })

    canonical_idx = next(
        i for i, r in enumerate(sweep)
        if abs(r["obstacle_density"] - CANONICAL_DENSITY) < 1e-6
    )

    return {
        "version": 1,
        "grid_size": int(N),
        "heuristic": HEURISTIC,
        "num_grids": int(B),
        "canonical_density_index": int(canonical_idx),
        "sweep": sweep,
    }


# -----------------------------------------------------------------------------
# Reference model (smoke test): uniform attention over free cells
# -----------------------------------------------------------------------------
def random_model_fn() -> ModelFn:
    """Return a ModelFn (takes grids, returns attention) emitting uniform
    attention over free cells. Pure NumPy; no torch, no GPU."""

    def _uniform_fn(grids: np.ndarray) -> np.ndarray:
        grids = np.asarray(grids)
        obstacle = grids[:, :, :, 0] > 0.5            # [B, N, N]
        attn = (~obstacle).astype(np.float32)
        sums = attn.sum(axis=(1, 2), keepdims=True)
        sums = np.where(sums > 0, sums, 1.0)
        return attn / sums

    return _uniform_fn
