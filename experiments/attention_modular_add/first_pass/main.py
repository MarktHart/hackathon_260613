"""First-pass attempt: hand-built Fourier/circular representation for modular addition.

The goal's CURRENT contract (../task.py) is:

    model_fn(tokens: np.ndarray [p*p, 3] int) -> (Q, K)
        with Q, K each float32 of shape [p*p, 3, d_head] (d_head = 128).

`evaluate` reads the query at the a-position (Q[:, 0, :]) and the key at the
b-position (K[:, 1, :]) and sweeps every Fourier frequency k, measuring how the
freq-k subspaces of Q_a and K_b align. The clean Nanda-et-al. mechanism --
numbers as complex exponentials, addition as angle addition -- shows up as a
strong single-frequency alignment.

We implement that directly: embed each token value x as the Fourier feature
vector [sin(2pi k x / p), cos(2pi k x / p)]_k. The a-position query uses the
standard pattern; the b-position key uses the same pattern so the two freq-k
2D subspaces coincide, maximising alignment. All compute runs in torch on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

P = 97          # canonical prime modulus
D_HEAD = 128    # canonical head dimension
N_FREQ = min(D_HEAD // 2, P // 2)   # frequency pairs that fit in d_head (48 for p=97)


def _fourier_basis_t(p: int, d_head: int) -> torch.Tensor:
    """[p, d_head] Fourier basis on CUDA: cols (2k-2, 2k-1) = (sin_k, cos_k)."""
    x = torch.arange(p, dtype=torch.float32, device=DEVICE)          # (p,)
    k = torch.arange(1, N_FREQ + 1, dtype=torch.float32, device=DEVICE)  # (N_FREQ,)
    ang = 2.0 * np.pi * k[None, :] * x[:, None] / p                  # (p, N_FREQ)
    basis = torch.zeros((p, d_head), dtype=torch.float32, device=DEVICE)
    basis[:, 0:2 * N_FREQ:2] = torch.sin(ang)
    basis[:, 1:2 * N_FREQ:2] = torch.cos(ang)
    return basis


def fourier_model_fn(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Q, K each [batch, 3, d_head] implementing the Fourier head."""
    tokens = np.asarray(tokens)
    batch_size = tokens.shape[0]

    a_t = torch.as_tensor(tokens[:, 0], dtype=torch.long, device=DEVICE)
    b_t = torch.as_tensor(tokens[:, 1], dtype=torch.long, device=DEVICE)

    basis = _fourier_basis_t(P, D_HEAD)   # (P, D_HEAD)

    Q_a = basis[a_t]   # (batch, D_HEAD) Fourier features of a
    K_b = basis[b_t]   # (batch, D_HEAD) Fourier features of b (same subspaces)

    # Separator position: tiny constant so it carries no structure.
    sep = torch.full((batch_size, D_HEAD), 1e-3, dtype=torch.float32, device=DEVICE)

    Q = torch.stack([Q_a, K_b, sep], dim=1)   # (batch, 3, D_HEAD)
    K = torch.stack([Q_a, K_b, sep], dim=1)   # (batch, 3, D_HEAD)

    return Q.detach().cpu().numpy(), K.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(fourier_model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # Print headline values present in the evaluate payload.
    print(f"modulus={payload['modulus']} d_head={payload['d_head']}")
    print(f"max_alignment={payload['max_alignment']:.4f} "
          f"argmax_alignment_freq={payload['argmax_alignment_freq']} "
          f"total_explained_variance={payload['total_explained_variance']:.4f}")
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
