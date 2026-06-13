"""attention_linear_sum / pass_3  — hand-built linear-attention head.

Hypothesis
----------
A single attention head can faithfully compute  y = α·x₁ + β·x₂  broadcast to
every target position, IF we drop the softmax normalisation (linear attention).

Mechanism (a minimal delta from experiments/base_model.py's `Attention`):
  - We hand-set the Q/K/V/O projections (no training).
  - The coefficient token (α, β) lives at position 2 and is *broadcast* into the
    residual stream of every target position t≥3 — i.e. it is supplied to the
    QUERY projection, exactly as the goal frames it ("coefficients supplied only
    in the query/key projections"). A trivial preceding copy/induction head would
    realise this broadcast in a real model; here we set it directly.
  - Q reads (α, β) from the residual.            q_t = [α, β]
  - K reads position identity.                   k_0 = [1, 0],  k_1 = [0, 1]
  - Score(t, j) = q_t · k_j   (LINEAR, no softmax)  →  α at j=0, β at j=1, 0 else.
  - V carries the scalar feature.                v_0 = x₁,  v_1 = x₂,  else 0.
  - out_t = Σ_j Score(t,j)·v_j = α·x₁ + β·x₂.   Exact for every (α, β).

The softmax variant of the SAME head is run as an explicit strawman: softmax
weights are non-negative and sum to 1, so they cannot express coefficients with
magnitude > 1 or with negative sign. That comparison is saved for the Demo tab.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda"

# task.py does `from benchmark import VERSION`; both live in the goal dir.
# load_task() execs task.py but does not add the goal dir to sys.path, so we do
# it here before importing/loading the task.
GOAL_DIR = Path(__file__).resolve().parent.parent
if str(GOAL_DIR) not in sys.path:
    sys.path.insert(0, str(GOAL_DIR))

from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)
Batch = task.Batch

D_MODEL = 32
T = 8

# Residual-stream channel layout (only a handful of the 32 dims are used).
C_X1, C_X2, C_A, C_B, C_ID0, C_ID1 = 0, 1, 2, 3, 4, 5


def _embed(batch):
    """Construct the residual stream (B, T, D_MODEL) from the batch fields."""
    B = batch.x1.shape[0]
    R = np.zeros((B, T, D_MODEL), dtype=np.float64)
    R[:, 0, C_X1] = batch.x1[:, 0]      # feature 1 at pos 0
    R[:, 1, C_X2] = batch.x2[:, 0]      # feature 2 at pos 1
    R[:, 0, C_ID0] = 1.0                # position-0 key marker
    R[:, 1, C_ID1] = 1.0                # position-1 key marker
    R[:, 2, C_A] = batch.alpha[:, 0]    # coefficient token at pos 2
    R[:, 2, C_B] = batch.beta[:, 0]
    # Broadcast the coefficient token into every target position's query slot.
    for t in range(3, T):
        R[:, t, C_A] = batch.alpha[:, 0]
        R[:, t, C_B] = batch.beta[:, 0]
    return R


def _projections():
    Wq = np.zeros((D_MODEL, 2)); Wq[C_A, 0] = 1.0; Wq[C_B, 1] = 1.0
    Wk = np.zeros((D_MODEL, 2)); Wk[C_ID0, 0] = 1.0; Wk[C_ID1, 1] = 1.0
    Wv = np.zeros((D_MODEL, 1)); Wv[C_X1, 0] = 1.0; Wv[C_X2, 0] = 1.0
    Wo = np.ones((1, 1))  # identity scalar read-out
    return Wq, Wk, Wv, Wo


def make_model_fn(use_softmax=False, temperature=1.0):
    Wq, Wk, Wv, Wo = _projections()

    Wq_t = torch.as_tensor(Wq, dtype=torch.float64, device=DEVICE)
    Wk_t = torch.as_tensor(Wk, dtype=torch.float64, device=DEVICE)
    Wv_t = torch.as_tensor(Wv, dtype=torch.float64, device=DEVICE)
    Wo_t = torch.as_tensor(Wo, dtype=torch.float64, device=DEVICE)

    def fn(batch):
        R = torch.as_tensor(_embed(batch), dtype=torch.float64, device=DEVICE)
        Q = R @ Wq_t                        # (B, T, 2)
        K = R @ Wk_t                        # (B, T, 2)
        V = R @ Wv_t                        # (B, T, 1)
        scores = Q @ K.transpose(1, 2)      # (B, T, T)
        causal = torch.tril(torch.ones((T, T), dtype=torch.float64, device=DEVICE))
        if use_softmax:
            s = scores / temperature
            s = torch.where(causal[None] > 0, s, torch.full_like(s, -1e9))
            w = torch.softmax(s, dim=-1)
        else:
            w = scores * causal[None]       # linear attention, causal-masked
        out = (w @ V) @ Wo_t                # (B, T, 1)
        return out[:, 3:8, 0].detach().cpu().numpy().astype(np.float32)  # (B, 5)

    return fn


def _sweep_r2(model_fn):
    """Re-run the goal's 24-pair sweep to capture per-pair R² for the viz."""
    grid = {}
    for a, b in task._sweep_coeffs():
        rng = np.random.default_rng(123)
        B = 256
        x1 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
        x2 = rng.uniform(-1, 1, size=(B, 1)).astype(np.float32)
        bb = Batch(x1=x1, x2=x2,
                   alpha=np.full((B, 1), a, np.float32),
                   beta=np.full((B, 1), b, np.float32))
        pred = model_fn(bb)
        tgt = np.repeat(a * x1 + b * x2, 5, axis=1)
        mse = float(np.mean((pred - tgt) ** 2))
        var = float(np.var(tgt))
        r2 = 1.0 - mse / var if var > 0 else 0.0
        grid[f"{a},{b}"] = {"alpha": a, "beta": b, "r2": r2, "mse": mse}
    return grid


def main():
    run_dir = results_dir(__file__)

    linear_fn = make_model_fn(use_softmax=False)
    softmax_fn = make_model_fn(use_softmax=True, temperature=1.0)

    # Headline: the linear-attention head is the attempt's contribution.
    payload = task.evaluate(linear_fn)
    record_benchmark(__file__, run_dir, payload)

    # Strawman comparison + sweep heatmaps for the Demo tab.
    artefact = {
        "sweep_linear": _sweep_r2(linear_fn),
        "sweep_softmax": _sweep_r2(softmax_fn),
        "canonical_pred": payload["canonical"]["pred"],
        "canonical_target": payload["canonical"]["target"],
        "config": payload["config"],
    }
    (run_dir / "viz.json").write_text(json.dumps(artefact))

    pred = np.array(payload["canonical"]["pred"])
    tgt = np.array(payload["canonical"]["target"])
    r2 = 1.0 - np.mean((pred - tgt) ** 2) / np.var(tgt)
    print(f"[pass_3] linear-attention head  R2_canonical = {r2:.6f}")
    print(f"[pass_3] artefacts -> {run_dir}")


if __name__ == "__main__":
    main()
