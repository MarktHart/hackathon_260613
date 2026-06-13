import torch
import numpy as np
from agentic.experiments import load_task, record_benchmark, results_dir

task = load_task(__file__)

DEVICE = "cuda"

# Hand-built sign-threshold circuit: directly compute dot(q,k) and apply a sharp sigmoid
# centred at 0. This implements the idealised "sign detector" hypothesis.
# No training, no extra parameters — the mechanism is explicit and faithful.

SHARPNESS = 50.0  # temperature controlling transition steepness at dot = 0


def model_fn(queries_np: np.ndarray, keys_np: np.ndarray) -> np.ndarray:
    """
    queries: (n_pairs, d_model) float32
    keys:    (n_pairs, d_model) float32
    returns: (n_pairs,) float32 -- attention weight in [0, 1]
    """
    q = torch.as_tensor(queries_np, dtype=torch.float32, device=DEVICE)
    k = torch.as_tensor(keys_np, dtype=torch.float32, device=DEVICE)

    # Dot product = cosine similarity (inputs are unit-normalised by task generator)
    dot = torch.einsum("bd,bd->b", q, k)  # (n_pairs,)

    # Sharp sigmoid centred at 0: attention ≈ 1 for cos > 0, ≈ 0 for cos < 0
    with torch.no_grad():
        attn = torch.sigmoid(SHARPNESS * dot)

    return attn.detach().cpu().numpy().astype(np.float32)


payload = task.evaluate(model_fn)
payload["model_info"] = {
    "name": "hand_built_sign_threshold",
    "type": "heuristic",
    "notes": f"Direct dot-product sign detector with sigmoid sharpness={SHARPNESS}. No learning — pure circuit implementation of the hypothesised mechanism."
}

run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")
print(f"sign_sharpness_canonical: {payload.get('sweep', [{}])[0].get('mean_attention', 'N/A')}")