"""Fourth pass at attention_modular_add: a clean, GPU-aware hand-coded Fourier attention head.

This attempt implements the exact synthetic mechanism described in the goal's README and
validated in prior literature (Nanda et al., 2023; Zhong et al., 2023). The mechanism:

- Query at the 'a' position carries frequency vectors [sin(2πk a/p), cos(2πk a/p)] for
  k = 1..48 (p//2), interleaved across the 128-channel head.
- Key at the 'b' position carries the *conjugate* pattern: sine terms negated,
  cosine terms preserved, so the inner product Q(a)·K(b) = Σ_k cos(2πk(a+b)/p),
  which peaks when a + b ≡ const (mod p).
- The separator token '=' (id = p) is mapped to a tiny constant vector to avoid interference.

The model is fully deterministic, uses no training, and runs entirely on the GPU as required.
It satisfies the "hardcoded weights bonus" by expressing the mechanism as explicit
torch tensors on cuda.

Key implementation details:
- A single [p, d_head] Fourier basis is built once on CPU (float32), then moved to GPU.
- Token lookups (basis[a], basis[b]) and the conjugate-key negation (sine columns only)
  are performed on the GPU.
- Output Q, K are returned as NumPy arrays on CPU to satisfy the task.py contract.
"""

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; do not fall back to CPU

P = 97          # canonical prime modulus
D_HEAD = 128    # canonical head dimension
N_FREQ = P // 2 # 48 frequencies for p=97 (Nyquist)


def _fourier_basis(p: int, d_head: int = D_HEAD) -> np.ndarray:
    """
    Construct the Fourier basis matrix for token values in [0, p).

    Returns float32 array of shape (p, d_head) where for each k=1..p//2:
      - column 2*(k-1)   : sin(2π k x / p)   (query sine component)
      - column 2*(k-1)+1: cos(2π k x / p)   (query cosine component)
    Remaining columns (if d_head > 2*N_FREQ) filled with tiny noise.
    """
    n_freq = p // 2
    features = np.zeros((p, d_head), dtype=np.float32)
    x = np.arange(p, dtype=np.float32)
    for k in range(1, n_freq + 1):
        idx = 2 * (k - 1)
        features[:, idx] = np.sin(2 * np.pi * k * x / p)
        features[:, idx + 1] = np.cos(2 * np.pi * k * x / p)
    # Fill unused columns with tiny noise to preserve output shape
    if 2 * n_freq < d_head:
        noise = np.random.default_rng(42).uniform(-0.01, 0.01, size=(p, d_head - 2 * n_freq)).astype(np.float32)
        features[:, 2 * n_freq:] = noise
    return features


def fourier_head_model_fn(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Model function implementing the synthetic Fourier attention head.

    Args:
        tokens: int32 array of shape [batch_size, 3] with columns [a, b, p].

    Returns:
        Q, K: float32 arrays each of shape [batch_size, 3, d_head].
              Task only analyses Q[:, 0, :] (query at 'a') and K[:, 1, :] (key at 'b').
    """
    batch_size = tokens.shape[0]
    a_vals = tokens[:, 0]
    b_vals = tokens[:, 1]

    # Build full [p, D_HEAD] Fourier basis on CPU once.
    basis = _fourier_basis(P, D_HEAD)

    # Move to GPU and perform lookups there.
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=DEVICE)
    a_t = torch.as_tensor(a_vals, dtype=torch.long, device=DEVICE)
    b_t = torch.as_tensor(b_vals, dtype=torch.long, device=DEVICE)

    # Query at 'a' position: standard Fourier projection [sin, cos] for each k.
    Q_a = basis_t[a_t]  # [batch, D_HEAD]

    # Key at 'b' position: conjugate pattern [-sin, cos] for each k.
    # Negate the sine columns (even indices 0, 2, 4, ...) while preserving cosine columns.
    K_b = basis_t[b_t].clone()
    K_b[:, 0:2 * N_FREQ:2] = -K_b[:, 0:2 * N_FREQ:2]

    # Separator token '=' (position 2): small constant vector to avoid interference.
    small_const = 1e-3
    Q_sep = torch.full((batch_size, D_HEAD), small_const, dtype=torch.float32, device=DEVICE)
    K_sep = Q_sep.clone()

    # Stack into [batch, 3, D_HEAD]. Only positions 0 (Q) and 1 (K) are analysed.
    Q = torch.stack([Q_a, K_b, Q_sep], dim=1)
    K = torch.stack([Q_a, K_b, K_sep], dim=1)

    return Q.detach().cpu().numpy(), K.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(fourier_head_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    print(f"Headline payload (p={P}, d_head={D_HEAD}):")
    print(f"  modulus = {payload['modulus']}")
    print(f"  max_alignment = {payload['max_alignment']:.4f}")
    print(f"  argmax_alignment_freq = {payload['argmax_alignment_freq']}")
    print(f"  total_explained_variance = {payload['total_explained_variance']:.4f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()