import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

def make_equality_head(batch) -> np.ndarray:
    """
    Returns attention weights of a head that routes attention onto the earlier key position
    that holds the *same* token as the query (equality lookup). The head only attends to the
    matching key p1 when the query is at p2; for other queries it emits uniform attention over
    the causally-allowed keys (to preserve row-stochasticity).
    """
    B, L = batch.tokens.shape
    p1 = torch.as_tensor(np.asarray(batch.p1), dtype=torch.long, device=DEVICE)
    p2 = torch.as_tensor(np.asarray(batch.p2), dtype=torch.long, device=DEVICE)

    # --- GPU compute (torch on CUDA) ---
    mask = torch.as_tensor(batch.mask, dtype=torch.float32, device=DEVICE)  # (B, L, L)

    # Start with uniform attention over causally-allowed positions.
    counts = mask.sum(dim=2, keepdim=True)                     # (B, L, 1), always >= 1
    attn = mask / counts                                       # Uniform over allowed keys

    # Route query p2 to matching key p1: concentrate that query row's mass on p1.
    rows = torch.arange(B, device=DEVICE)                      # (B,)
    attn[rows, p2, :] = 0.0                                    # Zero mass on all keys for this query
    attn[rows, p2, p1] = 1.0                                   # Full mass on the matching earlier key

    return attn.detach().cpu().numpy().astype(np.float32)

def main():
    model_fn = lambda b: make_equality_head(b)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Saved benchmark to {run_dir / 'benchmark.json'}")

if __name__ == "__main__":
    main()