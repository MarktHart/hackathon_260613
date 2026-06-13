"""Data generation and evaluation for the attention_knapsack goal.

Exports
-------
generate(seed) -> Batch
    Deterministic batch of 0/1-knapsack instances with exact ground truth.
evaluate(model_fn) -> dict
    Runs an attempt's model_fn over the canonical batch + a capacity sweep
    and returns a payload dict consumed verbatim by benchmark.score().
random_model_fn() -> ModelFn
    Factory returning a callable with the model_fn signature whose body emits
    random selection probabilities. Pure NumPy. Used by the smoke test.

The model_fn contract
---------------------
    model_fn(weights, values, capacity) -> selection
        weights  : np.ndarray (batch_size, n_items) float32, item weights
        values   : np.ndarray (batch_size, n_items) float32, item values
        capacity : float, the shared knapsack capacity for the batch
        returns  : np.ndarray (batch_size, n_items) in [0, 1] — per-item
                   selection score/probability. Thresholded at 0.5 by evaluate.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Batch:
    weights: np.ndarray            # (batch_size, n_items) float32
    values: np.ndarray            # (batch_size, n_items) float32
    capacity: float               # scalar float
    optimal_selections: np.ndarray  # (batch_size, n_items) bool
    optimal_values: np.ndarray      # (batch_size,) float32


# ──────────────────────────────────────────────────────────────────────
# Exact knapsack solver (integer DP) — ground truth for small n_items
# ──────────────────────────────────────────────────────────────────────

def _solve_knapsack_exact(weights, values, capacity):
    """Exact 0/1 knapsack DP. weights/values are 1D arrays of length n_items."""
    n = len(weights)
    w_int = np.round(weights).astype(int)
    cap_int = int(np.floor(capacity))
    if cap_int < 0:
        cap_int = 0
    dp = np.full(cap_int + 1, -1.0, dtype=np.float64)
    dp[0] = 0.0
    take = np.zeros((n, cap_int + 1), dtype=bool)

    for i in range(n):
        wi = w_int[i]
        vi = float(values[i])
        if wi <= 0:
            wi = 1  # guard: treat degenerate zero-weight as weight 1
        for w in range(cap_int, wi - 1, -1):
            if dp[w - wi] >= 0:
                cand = dp[w - wi] + vi
                if cand > dp[w]:
                    dp[w] = cand
                    take[i, w] = True

    best_w = int(np.argmax(dp))
    best_val = float(dp[best_w])

    sel = np.zeros(n, dtype=bool)
    w = best_w
    for i in reversed(range(n)):
        if 0 <= w <= cap_int and take[i, w]:
            sel[i] = True
            w -= max(int(round(weights[i])), 1)
    return sel, best_val


def _batch_solve(weights, values, capacity):
    batch_size, n_items = weights.shape
    opt_sel = np.zeros((batch_size, n_items), dtype=bool)
    opt_val = np.zeros(batch_size, dtype=np.float32)
    for b in range(batch_size):
        sel, val = _solve_knapsack_exact(weights[b], values[b], capacity)
        opt_sel[b] = sel
        opt_val[b] = val
    return opt_sel, opt_val


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 256
N_ITEMS = 16
W_MAX = 10
V_MAX = 10
CANONICAL_CAPACITY_FRAC = 0.5
SWEEP_FRACS = (0.3, 0.4, 0.5, 0.6, 0.7)


def generate(
    seed: int = 0,
    batch_size: int = BATCH_SIZE,
    n_items: int = N_ITEMS,
    w_max: int = W_MAX,
    v_max: int = V_MAX,
    capacity_frac: float = CANONICAL_CAPACITY_FRAC,
) -> Batch:
    """Generate a deterministic batch of knapsack instances for a given seed."""
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, w_max + 1, size=(batch_size, n_items)).astype(np.float32)
    values = rng.integers(1, v_max + 1, size=(batch_size, n_items)).astype(np.float32)

    expected_total_weight = n_items * (w_max + 1) / 2.0
    capacity = float(capacity_frac * expected_total_weight)

    opt_sel, opt_val = _batch_solve(weights, values, capacity)
    return Batch(
        weights=weights,
        values=values,
        capacity=capacity,
        optimal_selections=opt_sel,
        optimal_values=opt_val,
    )


def _measure(model_fn, batch: Batch) -> dict:
    """Run model_fn on one batch, return a record of pre-aggregated scalars.

    Feasibility is folded into value: an infeasible selection contributes 0
    value, so optimality_gap penalises capacity violations directly.
    """
    sel = np.asarray(model_fn(batch.weights, batch.values, batch.capacity))
    sel = np.clip(sel.astype(np.float32), 0.0, 1.0)
    if sel.shape != batch.weights.shape:
        raise ValueError(
            f"model_fn returned shape {sel.shape}, expected {batch.weights.shape}"
        )
    hard = sel >= 0.5
    m_w = np.sum(hard * batch.weights, axis=1)
    raw_v = np.sum(hard * batch.values, axis=1)
    feas = m_w <= batch.capacity + 1e-6
    eff_v = np.where(feas, raw_v, 0.0)

    opt_mean = float(np.mean(batch.optimal_values))
    model_value = float(np.mean(eff_v))
    gap = 1.0 - model_value / opt_mean if opt_mean > 0 else 1.0
    return {
        "capacity_frac": float(round(batch.capacity / (N_ITEMS * (W_MAX + 1) / 2.0), 6)),
        "optimal_value": opt_mean,
        "model_value": model_value,
        "model_weight": float(np.mean(m_w)),
        "feasible_rate": float(np.mean(feas)),
        "optimality_gap": float(gap),
    }


def evaluate(model_fn) -> dict:
    """Run model_fn over the canonical batch + capacity sweep, return payload.

    The payload is self-contained: the greedy linear baseline is measured here
    on the identical batches, so benchmark.score() needs no model or numpy.
    """
    greedy = greedy_baseline_fn

    canonical_batch = generate(seed=42, capacity_frac=CANONICAL_CAPACITY_FRAC)
    canonical = _measure(model_fn, canonical_batch)
    baseline_canonical = _measure(greedy, canonical_batch)

    sweep = []
    baseline_sweep = []
    for cf in SWEEP_FRACS:
        batch = generate(seed=42, capacity_frac=cf)
        sweep.append(_measure(model_fn, batch))
        baseline_sweep.append(_measure(greedy, batch))

    return {
        "version": 1,
        "config": {
            "batch_size": BATCH_SIZE,
            "n_items": N_ITEMS,
            "w_max": W_MAX,
            "v_max": V_MAX,
            "capacity_frac": CANONICAL_CAPACITY_FRAC,
            "sweep_fracs": list(SWEEP_FRACS),
        },
        "canonical": canonical,
        "sweep": sweep,
        "baseline_canonical": baseline_canonical,
        "baseline_sweep": baseline_sweep,
    }


def random_model_fn(seed: int = 12345):
    """Factory: return a model_fn that emits uniform random selection probs.

    Matches the model_fn signature exactly. Pure NumPy, no torch, no GPU.
    """
    rng = np.random.default_rng(seed)

    def model_fn(weights, values, capacity):
        return rng.random(size=np.asarray(weights).shape).astype(np.float32)

    return model_fn


def greedy_baseline_fn(weights, values, capacity):
    """Deterministic value/weight-ratio greedy baseline. Returns 0/1 float32."""
    weights = np.asarray(weights)
    values = np.asarray(values)
    batch_size, n_items = weights.shape
    selections = np.zeros_like(weights, dtype=np.float32)
    for b in range(batch_size):
        ratios = values[b] / (weights[b] + 1e-8)
        order = np.argsort(-ratios)
        w_accum = 0.0
        for idx in order:
            if w_accum + weights[b, idx] <= capacity + 1e-6:
                selections[b, idx] = 1.0
                w_accum += float(weights[b, idx])
    return selections
