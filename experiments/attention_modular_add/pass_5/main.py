"""pass_5 — vectorized hand-built Fourier attention head (GPU).

Mechanism (Nanda et al. 2023; Zhong et al. 2023): the query at token `a` carries
[sin(2πk a/p), cos(2πk a/p)] for k=1..48 interleaved across channels; the key at
token `b` carries the *conjugate* [-sin(2πk b/p), cos(2πk b/p)] on the SAME
channels. Their inner product is Σ_k cos(2πk(a+b)/p), peaking at a+b≡const (mod p).
Because Q and K use the same channel pair for each frequency, the Q-side and
K-side freq-k subspaces coincide -> alignment ≈ 1 and conjugate phase error ≈ 0
for every frequency.

attempt type: hand_built. Everything is computed as torch tensors on cuda; no
training. Fully vectorized (no Python loops in the hot path) to stay under budget.
"""

import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback

P = 97
D_HEAD = 128
N_FREQ = P // 2  # 48


def _basis(device) -> torch.Tensor:
    """[p, D_HEAD] Fourier basis. Cols (2k-2,2k-1)=(sin,cos) for k=1..48; rest 0."""
    x = torch.arange(P, dtype=torch.float32, device=device).unsqueeze(1)      # [p,1]
    k = torch.arange(1, N_FREQ + 1, dtype=torch.float32, device=device)       # [48]
    ang = 2.0 * np.pi * x * k / P                                             # [p,48]
    sin, cos = torch.sin(ang), torch.cos(ang)                                 # [p,48]
    basis = torch.zeros(P, D_HEAD, dtype=torch.float32, device=device)
    basis[:, 0:2 * N_FREQ:2] = sin
    basis[:, 1:2 * N_FREQ:2] = cos
    return basis


def fourier_head_model_fn(tokens: np.ndarray):
    a = torch.as_tensor(tokens[:, 0], dtype=torch.long, device=DEVICE)
    b = torch.as_tensor(tokens[:, 1], dtype=torch.long, device=DEVICE)

    basis = _basis(DEVICE)                       # [p, D_HEAD]
    sign = torch.ones(D_HEAD, dtype=torch.float32, device=DEVICE)
    sign[0:2 * N_FREQ:2] = -1.0                  # negate sine cols -> conjugate key

    Q_a = basis[a]                               # [batch, D_HEAD]
    K_b = basis[b] * sign                        # [batch, D_HEAD]
    sep = torch.full((a.shape[0], D_HEAD), 1e-3, dtype=torch.float32, device=DEVICE)

    Q = torch.stack([Q_a, K_b, sep], dim=1)      # only [:,0,:] is analysed
    K = torch.stack([Q_a, K_b, sep], dim=1)      # only [:,1,:] is analysed
    return Q.detach().cpu().numpy(), K.detach().cpu().numpy()


def main():
    task = load_task(__file__)
    payload = task.evaluate(fourier_head_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    print(f"max_alignment={payload['max_alignment']:.4f} "
          f"argmax_freq={payload['argmax_alignment_freq']} "
          f"total_ev={payload['total_explained_variance']:.4f}")
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
