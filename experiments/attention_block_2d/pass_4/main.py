"""attention_block_2d — pass_4

Geometric interpretability method (no peeking at the task's generators).

We recover the spatial attention pattern *purely from the matrix geometry*:

  1. Binarise the row-stochastic matrix with an absolute threshold (allowed
     keys carry mass >> the sub-eps noise floor).
  2. GLOBAL: a single index p whose ROW is fully attended (global token
     attends everyone) AND whose COLUMN is fully attended (everyone attends
     the global token).
  3. CAUSAL_2D: the binary mask equals the lower-triangular raster mask.
  4. LOCAL / DILATED: read the set of (dr, dc) key-minus-query displacements.
     The distinct positive offsets give the dilation (smallest step) and the
     window radius (number of steps). dilation == 1 -> contiguous "local";
     dilation  > 1 -> "dilated".

All real compute runs in torch on CUDA. The method never imports the task's
private pattern generators — it reads structure off the matrix alone.
"""

from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)

H, W = 8, 8
N = H * W
TAU = 0.4 / N  # absolute attended-mass threshold (noise floor is ~eps/k << this)

# ---- precomputed grid geometry, all on GPU -------------------------------
_idx = torch.arange(N, device=DEVICE)
_rows = (_idx // W).to(torch.int64)
_cols = (_idx % W).to(torch.int64)
# DR[i, j] = row(j) - row(i)  (displacement of KEY relative to QUERY)
_DR = (_rows[None, :] - _rows[:, None]).to(torch.float32)
_DC = (_cols[None, :] - _cols[:, None]).to(torch.float32)
_TRIL = torch.tril(torch.ones(N, N, device=DEVICE))


def _window_mask(window_size: int, dilation: int) -> torch.Tensor:
    """Ideal binary mask for a (window_size, dilation) square window, on GPU."""
    vals = torch.tensor(
        [a * dilation for a in range(-window_size, window_size + 1)],
        device=DEVICE,
        dtype=torch.float32,
    )
    dr_in = (_DR[..., None] == vals).any(dim=-1)
    dc_in = (_DC[..., None] == vals).any(dim=-1)
    return (dr_in & dc_in).to(torch.float32)


def model_fn(attn: np.ndarray) -> dict:
    """Classify a single (N, N) row-stochastic attention matrix by geometry."""
    A = torch.as_tensor(attn, dtype=torch.float32, device=DEVICE)
    total = A.sum()
    M = (A > TAU).to(torch.float32)            # attended-key mask
    rowcount = M.sum(dim=1)                     # keys attended per query
    colcount = M.sum(dim=0)                     # queries attending each key

    # -------- 1. GLOBAL: same index p has a full row AND a full column ----
    full_row = rowcount >= (N - 0.5)
    full_col = colcount >= (N - 0.5)
    both = (full_row & full_col).nonzero(as_tuple=False).flatten()
    if both.numel() > 0:
        p = int(both[0].item())
        Mhat = torch.zeros(N, N, device=DEVICE)
        Mhat[p, :] = 1.0
        Mhat[:, p] = 1.0
        conf = float(((A * Mhat).sum() / total).item())
        return {"pattern_id": "global", "params": {"global_pos": p},
                "confidence": conf}

    # -------- 2. CAUSAL_2D: mask matches lower-triangular raster ----------
    tril_match = float((M == _TRIL).to(torch.float32).mean().item())
    if tril_match > 0.97:
        conf = float(((A * _TRIL).sum() / total).item())
        return {"pattern_id": "causal_2d", "params": {}, "confidence": conf}

    # -------- 3. LOCAL / DILATED: read displacement offsets ---------------
    mb = M.bool()
    drv = _DR[mb].abs()
    dcv = _DC[mb].abs()
    offs = torch.cat([drv, dcv])
    offs = offs[offs > 0.5]
    if offs.numel() == 0:
        window_size, dilation = 1, 1            # degenerate: self-only
    else:
        pos = torch.unique(offs)               # sorted ascending
        dilation = int(round(float(pos[0].item())))
        window_size = int(pos.numel())

    family = "dilated" if dilation > 1 else "local"
    Mhat = _window_mask(window_size, dilation)
    conf = float(((A * Mhat).sum() / total).item())

    if family == "local":
        params = {"window_size": window_size}
    else:
        params = {"window_size": window_size, "dilation": dilation}
    return {"pattern_id": family, "params": params, "confidence": conf}


def main() -> None:
    run_dir = results_dir(__file__)
    print(f"[pass_4] writing results to {run_dir}")
    payload = task.evaluate(model_fn)
    n_correct = sum(1 for r in payload["sweep"] if r["correct"])
    print(f"[pass_4] canonical accuracy: {n_correct}/{len(payload['sweep'])}")
    record_benchmark(__file__, run_dir, payload)


if __name__ == "__main__":
    main()
