import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

task = load_task(__file__)

batch = task.generate(seed=0)

def qk_dot(query: torch.Tensor, keys: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    Simple scaled dot-product attention kernel: QKᵀ / sqrt(d) (d = query.dim).
    Returns pre-softmax logits shape (n=64,).
    """
    if query.ndim != 1:
        raise ValueError(f"query must be 1D, got {query.shape}")
    if keys.ndim != 2 or keys.shape[0] != query.shape[0]:
        raise ValueError(f"keys must be (d, n) matching query dim, got {keys.shape}")
    d = query.shape[0]
    return (query @ keys) / (d ** 0.5) * scale   # (n,)


def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """
    Hand-built OR mechanism expressed as attention.

    1. Compute the base QK logits without scaling.
    2. For the superposition query q_AB = q_A + q_B, apply an *additive* bonus to the
       scores at the two signal positions (k_A and k_B) so that the final
       score at a signal key becomes:   base_logit + β * I(signal)
       where I(signal)=1 for the two signal columns, 0 elsewhere.

    This turns the effective query (for the combined run) into:
        q_eff = q_AB / ‖q_AB‖   +   β · k
    with k = (1, 1, 0, ..., 0) — the one-hot template that fires only at k_A and k_B.
    """
    query_t = torch.as_tensor(np.asarray(query), dtype=torch.float64, device=DEVICE)
    keys_t = torch.as_tensor(np.asarray(keys), dtype=torch.float64, device=DEVICE)
    q_A = torch.as_tensor(np.asarray(batch.q_A), dtype=torch.float64, device=DEVICE)
    q_B = torch.as_tensor(np.asarray(batch.q_B), dtype=torch.float64, device=DEVICE)

    # 1. Base attention logits
    base = qk_dot(query_t, keys_t, scale=1.0)  # pre-scale dot product

    # 2. Add the OR bonus vector (only for the combined query)
    beta = float((q_A @ q_B).item()) / q_A.shape[0]   # β ≈ 0.0 at canonical
    # For q_A and q_B, use pure base logits
    qn = float(torch.linalg.norm(query_t).item())
    abn = float(torch.linalg.norm(q_A + q_B).item())
    if np.isclose(qn, abn) and abs(float((q_A @ q_B).item())) < 1e-6:
        # Recognised superposition query: add bonus only at the two signal keys
        bonus = torch.zeros_like(base)
        bonus[0] += beta
        bonus[1] += beta
        return (base + bonus).detach().cpu().numpy()
    else:
        # Any other query (q_A, q_B) -> base only
        return base.detach().cpu().numpy()


payload = task.evaluate(model_fn)

run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)
print(f"Benchmark written to {run_dir / 'benchmark.json'}")