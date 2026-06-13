from __future__ import annotations
import numpy as np
import torch
from typing import Callable

from agentic.experiments import load_task, record_benchmark
from agentic.experiments import results_dir as get_results_dir

DEVICE = "cuda"

task = load_task(__file__)

# Canonical grid (matches task.generate)
H, W = 8, 8
N = H * W


def _norm(A: np.ndarray) -> np.ndarray:
    """Deterministic row-normalisation (no noise) for clean reference matrices."""
    return A / A.sum(axis=1, keepdims=True)


def _candidates():
    """Candidate (pattern_id, params, normalised_matrix) tuples, built from the
    task's own generators so they are directly comparable to generated matrices."""
    cands = []
    for ws in (1, 2, 3):
        cands.append(("local", {"window_size": ws}, _norm(task._make_local(N, H, W, ws))))
    for ws, dil in ((1, 1), (1, 2), (2, 1)):
        cands.append(("dilated", {"window_size": ws, "dilation": dil},
                      _norm(task._make_dilated(N, H, W, ws, dil))))
    for gp in (0, 63):
        cands.append(("global", {"global_pos": gp}, _norm(task._make_global(N, gp))))
    cands.append(("causal_2d", {}, _norm(task._make_causal_2d(N))))
    return cands


_CANDS = _candidates()
_CAND_TENSOR = torch.as_tensor(
    np.stack([c[2] for c in _CANDS], axis=0), dtype=torch.float32, device=DEVICE
)


def model_fn(attn: np.ndarray) -> dict:
    """Classify a (N, N) attention matrix by nearest canonical pattern on the GPU."""
    at = torch.as_tensor(attn, dtype=torch.float32, device=DEVICE)
    diffs = (_CAND_TENSOR - at[None, :, :]).reshape(_CAND_TENSOR.shape[0], -1)
    dists = torch.linalg.norm(diffs, dim=1)
    best = int(torch.argmin(dists).item())
    best_dist = float(dists[best].item())

    pattern_id, params, _ = _CANDS[best]
    confidence = float(np.exp(-best_dist))
    return {"pattern_id": pattern_id, "params": dict(params), "confidence": confidence}


def main():
    run_path = get_results_dir(__file__)
    print(f"Writing results to {run_path}")
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_path, payload)


if __name__ == "__main__":
    main()
