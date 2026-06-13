import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
N_HEADS = 4


def model_fn(input_ids: np.ndarray) -> dict[str, np.ndarray]:
    """
    Hand-coded stack-like attention for Dyck-1 generation.
    For each closing parenthesis, attend strongly to its matching opening parenthesis.
    Runs the stack-matching and attention construction on CUDA.
    """
    tokens = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    batch, seq_len = tokens.shape

    # Attention tensor: [batch, n_heads, seq_len, seq_len]
    attention = torch.zeros((batch, N_HEADS, seq_len, seq_len), dtype=torch.float32, device=DEVICE)

    # Process each sequence independently to find matching pairs
    for b_idx in range(batch):
        seq = tokens[b_idx]
        stack = []
        pairs = []  # list of (open_pos, close_pos)

        # Single pass to find all matching pairs
        for pos in range(seq_len):
            tok = seq[pos].item()
            if tok == 1:  # OPEN
                stack.append(pos)
            elif tok == 2:  # CLOSE
                if stack:
                    open_pos = stack.pop()
                    pairs.append((open_pos, pos))

        # Build attention rows
        for i in range(seq_len):
            tok = seq[i].item()
            causal_mask = torch.arange(i + 1, device=DEVICE)  # positions 0..i inclusive

            if tok == 2:  # CLOSE token - find its matching OPEN
                match_open = None
                for op, cp in pairs:
                    if cp == i:
                        match_open = op
                        break

                if match_open is not None:
                    # Strong peak on matching open (0.7 mass)
                    attention[b_idx, :, i, match_open] = 0.7
                    # Remaining 0.3 distributed uniformly over causal prefix
                    remaining = 0.3 / len(causal_mask)
                    attention[b_idx, :, i, causal_mask] = remaining
                    # Reinforce the match position
                    attention[b_idx, :, i, match_open] = 0.7
                else:
                    # Unmatched close (should not happen in valid Dyck-1): uniform causal
                    uniform_val = 1.0 / len(causal_mask)
                    attention[b_idx, :, i, causal_mask] = uniform_val

            else:  # OPEN or PAD - uniform causal attention
                uniform_val = 1.0 / len(causal_mask)
                attention[b_idx, :, i, causal_mask] = uniform_val

    # Normalize rows to sum to 1 (guard against numerical drift)
    row_sums = attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    attention = attention / row_sums

    return {"attention": attention.detach().cpu().numpy().astype(np.float32)}


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)


if __name__ == "__main__":
    main()