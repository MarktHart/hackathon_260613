"""Third pass at attention_modular_add: a clean, GPU-aware implementation of the synthetic Fourier attention head.

The mechanism is the same as pass_2 but corrected for broadcasting errors. The key fixes are:

1. **Broadcast shape in _fourier_basis** — we build a [p, D_HEAD] matrix by allocating a float32 array of shape (P, D_HEAD) and then, for each k, compute both sin and cos terms on the same row vector using vectorised NumPy. The sin terms are placed on even columns, cos on odd columns; any unused trailing columns are filled with tiny noise.

2. **GPU movement only after constructing NumPy basis** — we assemble the [p, d_head] numpy Fourier basis in CPU, then convert the entire batch of token lookups onto the GPU using torch.as_tensor with device=DEVICE. This respects the NumPy → GPU contract in task.py.

3. **Conjugate key via channel negation** — the query side uses the standard Fourier pattern [sin, cos], the key side uses the pattern for addition via the identity cos θ cos φ − sin θ sin φ = cos(θ+φ). We achieve this by negating the odd-indexed columns (sin terms) of the key side only, leaving the even-indexed columns (cos terms) unchanged.

4. **Deterministic separator projection** — both query and key at the '=' token are mapped to a tiny constant vector to avoid interfering with the mechanism, matching the baseline random_model_fn's behavior.

The implementation is a pure hand-coded circuit — no neural network, no training — satisfying the "hardcoded weights bonus". It satisfies the contract: input tokens of shape [batch_size, 3] int32, output [batch_size, 3, d_head] float32.

The only tunable parameter is N_FREQ = min(64, P//2) = 48, the number of full-frequency pairs used.

The model_fn runs entirely on the GPU as required by the pipeline: NumPy inputs are wrapped in torch tensors, and the Fourier lookup and conjugate-key negation are performed on Device="cuda".
"""

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # the pipeline guarantees a GPU is visible; do not fall back to CPU

P = 97          # canonical prime modulus
D_HEAD = 128    # canonical head dimension
N_FREQ = min(64, P // 2)   # use up to 64 frequencies, capped by Nyquist (48 for p=97)


def _fourier_basis(p: int, d_head: int = D_HEAD) -> np.ndarray:
    """
    Construct the Fourier basis matrix for token values in [0, p).

    Returns a float32 array of shape (p, d_head) where:
      - For k = 1, 2, ..., N_FREQ we place:
          column 2*(k-1)   : sin(2π k x / p)   (real part, Q-side)
          column 2*(k-1)+1: cos(2π k x / p)   (imag part, Q-side)
      - The remaining columns (if d_head > 2*N_FREQ) are filled with tiny random noise
        to preserve the required output shape.
    """
    features = np.zeros((p, d_head), dtype=np.float32)
    x = np.arange(p, dtype=np.float32)          # token value axis
    for k in range(1, N_FREQ + 1):
        idx = 2 * (k - 1)                        # start column of this frequency pair
        features[:, idx] = np.sin(2 * np.pi * k * x / p)   # sin component for query
        features[:, idx + 1] = np.cos(2 * np.pi * k * x / p)  # cos component for query
    # Fill unused columns (if d_head > 2*N_FREQ) with tiny noise
    if 2 * N_FREQ < d_head:
        remaining = d_head - 2 * N_FREQ
        for i in range(p):
            noise = np.random.uniform(-0.01, 0.01, size=remaining)
            features[i, 2 * N_FREQ:2 * N_FREQ + remaining] = noise
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
    a = tokens[:, 0]  # token value a
    b = tokens[:, 1]  # token value b
    p = tokens[:, 2]  # separator token id, same as the prime modulus

    # Build the full [p, D_HEAD] Fourier basis on the CPU once.
    basis = _fourier_basis(P, D_HEAD)

    # Move the basis and token index tensors onto the GPU and perform all lookups there.
    basis_t = torch.as_tensor(basis, dtype=torch.float32, device=DEVICE)
    a_t = torch.as_tensor(a, dtype=torch.long, device=DEVICE)
    b_t = torch.as_tensor(b, dtype=torch.long, device=DEVICE)

    # For token a (query side): use the standard Fourier projection.
    Q_a = basis_t[a_t]   # shape [batch, D_HEAD]

    # For token b (key side): use the *conjugate* frequency pattern, yielding
    # inner product proportional to cos(2π k (a + b) / p) for addition.
    # This is achieved by negating the sine terms (odd-indexed columns) of the basis
    # while leaving the cosine terms unchanged.
    K_b = basis_t[b_t].clone()
    # flip the sign of sine components (even column indices in the [sin, cos] ordering)
    K_b[:, 0:2*N_FREQ:2] = -K_b[:, 0:2*N_FREQ:2]

    # For the separator token at index 2, set query and key to a small constant vector
    # to avoid interfering with the mechanism, matching the baseline random_model_fn.
    small_const = 1e-3
    Q_sep = torch.full((batch_size, D_HEAD), small_const, dtype=torch.float32, device=DEVICE)
    K_sep = Q_sep.clone()   # identical small vector for key

    # Build the [batch, 3, D_HEAD] tensors
    Q = torch.stack([Q_a, K_b, Q_sep], dim=1)   # query: at position 0 (a), at position 1 (b) we reuse K_b as query
    K = torch.stack([Q_a, K_b, K_sep], dim=1)   # key: at position 1 (b) is K_b; we reuse Q_a as key at a for symmetry

    # Return NumPy arrays on the CPU as required by task.py.
    return Q.detach().cpu().numpy(), K.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(fourier_head_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Print headline values present in the evaluate payload (the benchmark-only
    # metric keys like 'fourier_alignment_canonical' are produced by
    # benchmark.score, not by task.evaluate, so we don't index them here).
    print(f"Headline payload (p={P}, d_head={D_HEAD}):")
    print(f"  modulus = {payload['modulus']}")
    print(f"  max_alignment = {payload['max_alignment']:.4f}")
    print(f"  argmax_alignment_freq = {payload['argmax_alignment_freq']}")
    print(f"  total_explained_variance = {payload['total_explained_variance']:.4f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()