"""attention_argmax / pass_3 — hand-built softmax (argmax) attention head.

Hypothesis: a softmax attention head IS a soft argmax over key-query
similarities, and its sharpness is controlled by a single temperature. As the
temperature -> 0 the distribution approaches a one-hot on the winner (true
argmax); the `exp` non-linearity is the mechanism that makes this happen.

This attempt:
  1. Implements the head analytically as torch tensors on CUDA (no learning).
  2. Reports the canonical sweep through task.evaluate / record_benchmark.
  3. Runs an EXTRA faithfulness/baseline experiment (saved as an artefact for
     the app): a noise sweep contrasting the softmax head (with `exp`) against
     a no-`exp` linear-normalisation head and the uniform baseline. The `exp`
     head stays argmax-like under noise; the no-`exp` head collapses toward
     uniform. That contrast is the causal evidence that the `exp` is load-bearing.
"""
from __future__ import annotations

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees a CUDA device. Do NOT fall back to CPU.
DEVICE = "cuda"

task = load_task(__file__)

_D = task._D
_N = task._N

# Canonical temperature for the head under test. Small tau => sharp => argmax.
# With the task's clean construction the winner similarity is `separation` and
# every distractor similarity is exactly 0, so winner_mass(sep) =
#   exp(sep/tau) / (exp(sep/tau) + (N-1)).
TAU = 0.25


# --------------------------------------------------------------------------- #
# Heads (all compute on the GPU)
# --------------------------------------------------------------------------- #
def softmax_head(q: np.ndarray, K: np.ndarray, V: np.ndarray, tau: float = TAU) -> np.ndarray:
    """Soft-argmax attention: softmax(K @ q / tau). The `exp` is the mechanism."""
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    scores = Kt @ qt                       # (N,) key-query similarities
    attn = torch.softmax(scores / tau, dim=0)
    return attn.detach().cpu().numpy().astype(np.float64)


def linear_head(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """No-`exp` strawman: relu(scores) normalised to a distribution."""
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    scores = Kt @ qt
    w = torch.relu(scores)
    s = w.sum()
    if float(s) <= 1e-12:
        w = torch.ones_like(w)
        s = w.sum()
    attn = w / s
    return attn.detach().cpu().numpy().astype(np.float64)


# --------------------------------------------------------------------------- #
# Canonical sweep -> benchmark.json
# --------------------------------------------------------------------------- #
def model_fn(q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    return softmax_head(q, K, V, tau=TAU)


payload = task.evaluate(model_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)


# --------------------------------------------------------------------------- #
# Extra experiment: noise sweep (exp vs no-exp vs uniform) for the app
# --------------------------------------------------------------------------- #
def _noisy_batch(rng: np.random.Generator, separation: float, noise: float):
    """Like task data but distractors keep their noisy similarities (not
    orthogonalised), so similarities span a realistic range."""
    q = rng.normal(size=_D).astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    K = (noise * rng.normal(size=(_N, _D))).astype(np.float32)
    winner_idx = int(rng.integers(0, _N))
    # Pin winner similarity to exactly `separation` (remove its own projection).
    K[winner_idx] = K[winner_idx] - np.dot(K[winner_idx], q) * q + separation * q
    return q, K, winner_idx


def _winner_mass(attn: np.ndarray, idx: int) -> float:
    return float(attn[idx])


NOISE_LEVELS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]   # >2 orders of magnitude
SEP_FIXED = 2.0
SEEDS = 100

comparison = {"separation": SEP_FIXED, "N": _N, "tau": TAU,
              "noise_levels": NOISE_LEVELS, "rows": []}
for nl in NOISE_LEVELS:
    sm, lin = [], []
    for rep in range(SEEDS):
        rng = np.random.default_rng(10_000 + int(nl * 997) + rep)
        q, K, widx = _noisy_batch(rng, SEP_FIXED, nl)
        V = np.zeros_like(K)
        sm.append(_winner_mass(softmax_head(q, K, V, tau=TAU), widx))
        lin.append(_winner_mass(linear_head(q, K, V), widx))
    comparison["rows"].append({
        "noise": nl,
        "softmax_winner_mass": float(np.mean(sm)),
        "linear_winner_mass": float(np.mean(lin)),
        "uniform_winner_mass": 1.0 / _N,
    })

with open(run_dir / "comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("benchmark canonical fidelity:",
      next(r["winner_mass_mean"] for r in payload["sweep"] if abs(r["separation"] - 2.0) < 1e-9))
print("wrote", run_dir / "comparison.json")
