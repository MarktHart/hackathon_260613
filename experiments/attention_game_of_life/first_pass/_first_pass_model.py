# This module holds the actual model function that runs on GPU.
# It must import torch, use a DEVICE = "cuda" tensor, and return NumPy.

import torch

DEVICE = "cuda"   # the pipeline guarantees a visible GPU


def model_fn(grids: np.ndarray) -> np.ndarray:
    """Single-attention GoL head + hard-coded survival/birth rule.

    grids: (B, H, W) current board in {0, 1} as float32.
    Returns: (B, H, W) float32 logits for alive-next. A positive logit predicts alive.
    """
    B, H, W = grids.shape

    # --- Device transfers ---------------------------------------------------
    grids_t = torch.as_tensor(grids, dtype=torch.float32, device=DEVICE)

    # --- Attention setup ----------------------------------------------------

    # Use a single attention head.
    # Query: 9-neighbor kernel (3×3 window around each cell).
    # Key: eight relative positions (+ a self-location key to anchor the count).
    # Value: 1.0 per live cell; attention is effectively neighbor-count.

    # (B, H, W) -> (B, (H*W), 9)
    #   for each token, flatten the 3×3 neighborhood into a vector of 9 slots.
    flat_9 = grids_t[:, None, :, :]  # (B, 1, H, W)
    pad = torch.nn.functional.pad(flat_9, (1, 1, 1, 1), mode="circular")  # wrap around
    neighborhoods = pad[:, :, 1:-1, 1:-1].reshape(B, H * W, 9)   # (B, HW, 9)

    # Query: same kernel as key, so qk^T is a correlation of 9-vectors.
    # This effectively treats each pixel as a query whose neighbors are keys.
    query = neighborhoods        # (B, HW, 9)   -> weight matrix that pulls nearby cells
    key = neighborhoods          # (B, HW, 9)

    # Value is just the live-cell mask.
    value = (grids_t > 0.0).float() * 1.0    # (B, H, W) => (B, HW, 1) for broadcasting
    value = value.unsqueeze(-1).expand(-1, -1, 9)  # (B, HW, 9) — each key gets the same live value

    # --- Attention ----------------------------------------------------------

    # Score: dot(q, k) => neighbor count per token.
    scores = torch.einsum("bqi,bkj->bqkj", query, key).mean(dim=2)   # (B, HW, 9) -> (B, HW)

    # Temperature: learnable scalar in [0.1, 10] as softmax temperature.
    # Start with a small fixed value; in a trained version this would be a parameter.
    temp = 1.0  # scalar
    # Softmax over the 9 neighbor positions.
    attn = torch.softmax(scores / temp, dim=-1)  # (B, HW, 9) but scores was scalar? Actually scores is sum, need to shape.

    # Corrected: reshape neighborhoods to match key shape.
    key_flat = key.view(B, H * W, 9)  # already flat in the above code — keep as-is.

    # Compute attention as Q @ K^T / temp, then normalise.
    # Q: (B, HW, 9)
    # K^T: (B, 9, HW) after transpose and reshape
    Q = query          # (B, HW, 9)
    K_T = key_flat.transpose(-2, -1)  # (B, 9, HW)
    att_logits = Q @ K_T / temp        # (B, HW, HW) -> neighbor count matrix

    # Sum over target positions to get total live neighbors for each query cell.
    # attn = torch.softmax(att_logits, dim=-1)  # (B, HW, HW)

    # Aggregate: sum the attended values.
    # attn = attn.view(B, H, W, 9).sum(dim=-1)   # (B, H, W) neighbor count
    counts = att_logits.view(B, H * W, H * W).sum(dim=-1)  # (B, H, W) neighbor count

    # --- GoL decision -------------------------------------------------------
    # alive_next = (current_alive and n == 2 or n == 3) or (not current_alive and n == 3)
    current_alive = (grids_t > 0.0).float()
    alive_next = torch.zeros_like(current_alive)  # (B, H, W)

    # Death: not alive and count not 3
    death = (current_alive == 0) & (counts != 3)
    # Survival: alive and count == 2 or 3
    survive = (current_alive == 1) & ((counts == 2) | (counts == 3))
    # Birth: not alive and count == 3
    birth = (current_alive == 0) & (counts == 3)
    alive_next = survive | birth

    # Convert to logits: we want a positive logit to mean alive.
    # We return a float32 array on CPU to satisfy task.expect_shape.
    logits = alive_next.float().detach().cpu().numpy()
    # For robustness, add a small bias so dead cells are not exactly 0.
    # In a trained version this would be the "no attention" bias.
    logits = logits - 0.1   # slight negative bias for dead cells
    return logits