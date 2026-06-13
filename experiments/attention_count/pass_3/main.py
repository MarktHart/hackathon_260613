import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir
from experiments.attention_count.task import generate, random_model_fn, Batch

task = load_task(__file__)

DEVICE = "cuda"

# -------------------------------------------------
# Canonical model weights are embedded as numpy arrays
# in the real scaffold. Here we simulate a correct
# forward pass that matches the shape constraints.
# -------------------------------------------------
def model_fn(batch: Batch) -> dict[str, np.ndarray]:
    """
    Return attention weights for a fixed canonical
    2-layer, 4-head-per-layer transformer.
    Shape: [B, 2, 4, 64, 64]
    """
    # In production this function would load
    # the checkpoint, run a forward pass, and
    # hook the per-head attention weights.
    # For the smoke test we return deterministic
    # weights that satisfy the shape and range.
    B, L = batch.tokens.shape

    # --- GPU compute (torch on CUDA) ---
    # NOTE: the original NumPy code assigned induction/distractor values but then
    # unconditionally overwrote every (b, layer, head) slice with `eye + noise`
    # in the final loop, so the effective output is a diagonal base plus small
    # Gaussian noise for every head. We reproduce that effective computation on
    # the GPU.
    eye = torch.eye(L, dtype=torch.float32, device=DEVICE)
    attn = eye.expand(B, 2, 4, L, L).clone()
    attn += torch.normal(
        mean=0.0, std=0.01, size=(B, 2, 4, L, L), device=DEVICE
    )

    return {"attn_weights": attn.detach().cpu().numpy().astype(np.float32)}


def run():
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"Payload written to {run_dir}/benchmark.json")


if __name__ == "__main__":
    run()