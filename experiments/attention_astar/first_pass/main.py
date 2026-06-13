import torch
import numpy as np

DEVICE = "cuda"

def model_fn(grid: np.ndarray, agent_pos: tuple[int, int], goal_pos: tuple[int, int]) -> np.ndarray:
    """
    Implementation of the hand-coded Attention A* circuit:
    f = g + h, where g = path length from agent to candidate (distance from center offset)
            h = Manhattan distance from candidate to goal

    Logits: (center_offset + manhattan_to_goal) * attention_kernel
    Uses attentionKernel = 1.0, bias = 1.0 for numeric stability.
    """
    H, W = grid.shape
    grid_t = torch.as_tensor(grid, dtype=torch.float32, device=DEVICE)  # (H, W)
    r0, c0 = agent_pos
    r1, c1 = goal_pos

    # Build center offset (distance from candidate to agent)
    offsets = torch.tensor([
        [-1, -1], [-1, 0], [-1, 1],
        [0, -1],  [0, 0],  [0, 1],
        [1, -1],  [1, 0],  [1, 1]
    ], dtype=torch.float32, device=DEVICE)  # 9 x 2
    # Compute center_offset = Manhattan distance from candidate to agent
    # (center = 0, edges = 2 each)
    center_offset = torch.abs(offsets).sum(-1)  # 9,
    # Compute Manhattan distance from candidate to goal
    # Use broadcasting across candidates to compute r1, c1 - offsets_t
    goal_t = torch.tensor([r1, c1], device=DEVICE, dtype=torch.float32)
    offset_t = torch.vstack([0.0, offsets])  # 10 x 2 (add zero for agent)
    r_goal = goal_t[0] - torch.arange(H, device=DEVICE).unsqueeze(-1)  # H x 10
    c_goal = goal_t[1] - torch.arange(W, device=DEVICE).unsqueeze(-1)
    r_diff = torch.abs(r_goal - offset_t[:, 0]).unsqueeze(-1)  # H x 10 x 2
    c_diff = torch.abs(c_goal - offset_t[:, 1]).unsqueeze(-1)
    manhattan_to_goal = (r_diff + c_diff).sum(-1).unsqueeze(0)  # 10 x 2
    # Remove agent row (=1) since we mask it anyway
    manhattan_togoal = manhattan_to_goal[1:]  # 9,

    # Combine f = g + h
    logits = torch.where(
        grid_t == 1,                     # obstacle
        torch.tensor(-float('inf'), device=DEVICE),  # -inf
        center_offset + manhattan_to_goal  # 9,
    )  # logits shape: (9,)

    # Shift to keep numerics stable while preserving ordering
    logits = logits - logits.min() + 1e-5

    return logits.detach().cpu().numpy()


if __name__ == "__main__":
    from agentic.experiments import load_task, record_benchmark, results_dir

    task = load_task(__file__)
    results_dir_path = results_dir(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, results_dir_path, payload)