"""First-pass attempt: hand-built Game of Life circuit using convolution on GPU.

This implements the exact Game of Life update rule as a fixed (non-learned)
torch circuit: circular convolution counts the 8 neighbours, then the
birth/survival logic is applied to produce per-cell logits.
"""
import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees a visible GPU; do not fall back to CPU.
DEVICE = "cuda"

# 3x3 kernel that sums the eight neighbours (center = 0).
_NEIGHBOR_KERNEL = torch.tensor(
    [[[1.0, 1.0, 1.0],
      [1.0, 0.0, 1.0],
      [1.0, 1.0, 1.0]]],
    dtype=torch.float32,
    device=DEVICE,
).unsqueeze(0)  # (1, 1, 3, 3) for conv2d


def game_of_life_model_fn(grids: np.ndarray) -> np.ndarray:
    """Compute one step of Conway's Game of Life (toroidal) and return logits.

    Args:
        grids: float32 array of shape (B, H, W) with values in {0.0, 1.0}.

    Returns:
        float32 array of shape (B, H, W) — logit > 0 means predicted alive.
    """
    # (B, H, W) -> (B, 1, H, W) for conv2d
    x = torch.as_tensor(grids, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    # Circular padding + convolution counts 8 neighbours per cell.
    # pad=(left, right, top, bottom) = (1, 1, 1, 1)
    neighbor_counts = F.conv2d(
        F.pad(x, pad=(1, 1, 1, 1), mode="circular"),
        _NEIGHBOR_KERNEL,
    ).squeeze(1)  # back to (B, H, W)

    # Game of Life rules:
    #  - survive: alive & (neighbors == 2 or 3)
    #  - birth:   dead  & (neighbors == 3)
    alive = (x.squeeze(1) > 0.5)
    survive = alive & ((neighbor_counts == 2) | (neighbor_counts == 3))
    birth = (~alive) & (neighbor_counts == 3)
    next_alive = survive | birth

    # Logits: large positive for alive, large negative for dead.
    logits = torch.where(next_alive, torch.tensor(10.0, device=DEVICE), torch.tensor(-10.0, device=DEVICE))

    return logits.detach().cpu().numpy().astype(np.float32)


if __name__ == "__main__":
    TASK = load_task(__file__)
    payload = TASK.evaluate(game_of_life_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Benchmark written to {run_dir / 'benchmark.json'}")