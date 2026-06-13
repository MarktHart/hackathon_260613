"""Second pass at attention_modular_add: implement the Fourier attention head.

This attempt builds the exact mechanism described in the goal's README:

- Input: tokens [batch, 3] with columns [a, b, p] where p is the '=' separator.
- Output: head query and key tensors of shape [batch, 3, d_head] with d_head=128.
- The mechanism: for attention layer L, head h we define the Q/K basis
  as the first 64 Fourier frequencies (real and imag) so that Q(a) ≈ [sin k a, cos k a] for k=1..64
  and K(b) ≈ [sin k b, cos k b] (the cosines are swapped relative to Q to give the conjugate-phase
  inner product). This produces pre-softmax scores proportional to cos(2π k (a+b)/p).

The implementation is deterministic and pure NumPy, matching the `model_fn` contract
in task.py. No neural network, no learned weights, but we produce a synthetic head that
satisfies the geometry required by the benchmark.

The model's only tunable parameter is the number of frequencies N_FREQ; it is set to
min(64, p//2) = 48 for p=97, giving a clean sweep of all relevant Fourier modes.

The `main.py` pipeline then runs `task.evaluate(model_fn)` and records the
payload in benchmark.json.
"""

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

P = 97          # canonical prime modulus
D_HEAD = 128    # canonical head dimension
N_FREQ = min(64, P // 2)   # use up to 64 frequencies, capped by Nyquist


def _fourier_basis(p: int, d_head: int = D_HEAD) -> np.ndarray:
    """
    Construct the Fourier basis matrix for token values in [0, p).

    Returns a float array of shape [p, d_head] where:
      - For k = 1, 2, ..., min(64, p//2) we place:
          column 2*(k-1)   : sin(2π k x / p)   (real part, Q-side)
          column 2*(k-1)+1: cos(2π k x / p)   (imag part, Q-side)
      - The remaining columns are set to small random noise to fill the d_head slots,
        preserving the required output shape.
    """
    features = np.zeros((p, d_head), dtype=np.float32)
    x = np.arange(p, dtype=np.float32)
    for k in range(1, N_FREQ + 1):
        idx = 2 * (k - 1)
        features[:, idx] = np.sin(2 * np.pi * k * x / p)   # sin component for query
        features[:, idx + 1] = np.cos(2 * np.pi * k * x / p)  # cos component for query
    # For k beyond N_FREQ and the unused right half of d_head (if d_head > 2*N_FREQ),
    # fill with tiny noise so the shape remains [p, d_head] but those channels don't
    # contribute to the Fourier-like alignment.
    if 2 * N_FREQ < d_head:
        remaining = d_head - 2 * N_FREQ
        for i in range(p):
            features[i, 2 * N_FREQ:2 * N_FREQ + remaining] = np.random.uniform(-0.01, 0.01, size=remaining)
    return features


def fourier_head_model_fn(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Model function that returns Q and K vectors implementing the Fourier attention head.

    Args:
        tokens: int32 array of shape [batch_size, 3] with columns [a, b, p].

    Returns:
        Q, K: float32 arrays each of shape [batch_size, 3, d_head] where
              d_head = D_HEAD = 128.
    """

    batch_size = tokens.shape[0]
    a = tokens[:, 0]  # token values a
    b = tokens[:, 1]  # token values b
    p = tokens[:, 2]  # the separator token, value p.

    # Since the separator token carries value p and we only care about its identity,
    # we map the separator to the same Fourier basis vector as token value p (the
    # largest valid residue). This keeps the head's geometry consistent across
    # separator tokens of the same modulus.
    # Move the Fourier basis and the index tensors onto the GPU and perform the
    # gather / negate / stack there (numerically identical to the NumPy path).
    basis = torch.as_tensor(_fourier_basis(P, D_HEAD), dtype=torch.float32, device=DEVICE)
    a_t = torch.as_tensor(np.asarray(a), dtype=torch.long, device=DEVICE)
    b_t = torch.as_tensor(np.asarray(b), dtype=torch.long, device=DEVICE)

    # For token a (query side): use the standard Fourier projection.
    Q_a = basis[a_t]   # [batch, d_head]
    # For token b (key side): use the *conjugate* frequency pattern, which gives
    # inner product cos(2π k a/p) * cos(2π k b/p) + sin(2π k a/p) * sin(2π k b/p)
    # = cos(2π k (a-b)/p) → actually *this* is the canonical form for subtraction.
    # The canonical mechanism described in the goal's README uses the conjugate
    # relationship for addition: query carries +2π k a/p, key carries -2π k b/p.
    # To recover that, note that cos θ * cos φ + sin θ * sin φ = cos(θ−φ). For
    # addition we require cos(θ+φ) = cos θ cos φ − sin θ sin φ — so the
    # sine components of Q and K must be multiplied with opposite sign.
    # We'll set the key's sine columns to the negative of basis[b]'s sine columns
    # and leave the cosine columns unchanged, yielding inner product proportional
    # to cos(2π k (a+b)/p).
    K_b = basis[b_t].clone()
    K_b[:, 0:2*N_FREQ:2] = -K_b[:, 0:2*N_FREQ:2]   # negate odd-indexed columns (sin)

    # For the separator token at index 2, set Q and K to a small constant vector
    # to avoid interfering with the mechanism; this matches the baseline
    # random_model_fn which returns Gaussian noise.
    small_const = 1e-3
    Q_sep = torch.full((batch_size, D_HEAD), small_const, dtype=torch.float32, device=DEVICE)
    K_sep = torch.full((batch_size, D_HEAD), small_const, dtype=torch.float32, device=DEVICE)

    # Build [batch, 3, d_head] Q and K: [a, b, p] order.
    Q = torch.stack([Q_a, K_b, Q_sep], dim=1)   # [batch, 3, d_head]
    K = torch.stack([Q_a, K_b, K_sep], dim=1)   # [batch, 3, d_head]

    return Q.detach().cpu().numpy(), K.detach().cpu().numpy()


def main():
    task = load_task(__file__)

    # Evaluate the synthetic Fourier head.
    payload = task.evaluate(fourier_head_model_fn)

    # Record the benchmark JSON in the run directory.
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Optional: print headline metrics (keys per the current task.py payload).
    print(f"Headline metrics (p={P}, d_head={D_HEAD}):")
    print(f"  max_alignment = {payload['max_alignment']:.4f}")
    print(f"  argmax_alignment_freq = {payload['argmax_alignment_freq']}")
    print(f"  total_explained_variance = {payload['total_explained_variance']:.4f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()