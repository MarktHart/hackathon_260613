import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# -------------------------------------------------
# model_fn: maps a Batch -> attention weights (n_sequences, seq_len, seq_len)
# signature required by task.evaluate (see task.py)
# -------------------------------------------------
def model_fn(batch) -> np.ndarray:
    """
    Hand-built wildcard matcher: the query at the target position attends to
    the anchor token at position 0 (the wildcard span is skipped). Every other
    query row uses uniform attention.

    Returns attention weights of shape (n_sequences, seq_len, seq_len).
    """
    sequences = torch.as_tensor(batch.sequences, device=DEVICE)
    n_seq, L = sequences.shape
    target_pos = int(batch.target_pos)
    anchor_pos = int(batch.anchor_pos)

    # Start from uniform attention so every row is a valid distribution.
    attn = torch.full((n_seq, L, L), 1.0 / L, dtype=torch.float32, device=DEVICE)

    # Override the target query row: attend fully to the anchor at position 0.
    attn[:, target_pos, :] = 0.0
    attn[:, target_pos, anchor_pos] = 1.0

    return attn.detach().cpu().numpy().astype(np.float32)

def run():
    task = load_task(__file__)
    run_dir = results_dir(__file__)
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

if __name__ == "__main__":
    run()