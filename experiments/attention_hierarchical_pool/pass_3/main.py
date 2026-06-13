"""attention_hierarchical_pool / pass_3 — hand_built.

A genuine attention-only head whose effective score matrix is produced by
HAND-SET Q/K projection weights acting on positional features. The weights
realise a Gaussian distance kernel

    score_ij = -(i - j)^2 / (2 * sigma_L^2)   (within the query's chunk, else -inf)

so softmax over keys gives attention centred on the query with standard
deviation sigma_L. The ONLY thing that changes with depth is sigma_L, which
grows geometrically across the 12 layers: early layers pool a tight local
neighbourhood (small sigma), late layers pool the whole 16-token chunk
(large sigma). This is exactly the fine -> coarse "hierarchical pooling"
signature the goal asks about, and it is computed as real QK attention on the
GPU, not a hand-painted indicator matrix.

Delta from base_model.py: a single attention layer, no MLP, no residual; the
QKV projections are hand-set (not learned) and positional rather than from a
token embedding, and sigma is indexed by layer so the receptive field widens
with depth. Everything below runs in torch on CUDA.
"""

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

task = load_task(__file__)

# Canonical constants (mirror task.py)
SEQ_LEN = 256
NUM_LAYERS = 12
NUM_HEADS = 8
CHUNK_SIZE = 16
NUM_CHUNKS = 16

# Receptive-field width schedule. sigma grows geometrically from a sub-token
# local window (early layers) to roughly the chunk scale (late layers).
SIGMA_MIN = 0.55   # layer 0 base width  -> essentially attends to the query token
SIGMA_MAX = 7.0    # layer 11 base width -> spreads across the 16-token chunk

# Precompute positional feature matrix and chunk mask once, on the GPU.
# Feature for position p: f(p) = [p, p^2, 1]. These are fixed positional
# encodings (the "base model" positional features); the head-specific circuit
# lives entirely in the hand-set projection weights below.
_pos = torch.arange(SEQ_LEN, dtype=torch.float32, device=DEVICE)
_P = torch.stack([_pos, _pos * _pos, torch.ones_like(_pos)], dim=1)  # (256, 3)
_chunk_id = (_pos // CHUNK_SIZE).long()                              # (256,)
_same_chunk = _chunk_id[:, None] == _chunk_id[None, :]              # (256, 256) bool

# Hand-set KEY projection: maps f(p) = [p, p^2, 1] -> k_p = [p, p^2].
# (Layer/head independent — the width lives in the query projection.)
_W_K = torch.tensor(
    [[1.0, 0.0],
     [0.0, 1.0],
     [0.0, 0.0]],
    dtype=torch.float32, device=DEVICE,
)
_K = _P @ _W_K  # (256, 2) rows = [j, j^2]


def _sigma(layer_idx: int, head_idx: int) -> float:
    """Per-(layer, head) Gaussian width. Base widens geometrically with depth;
    heads at a layer get a mild spread of widths so the layer is not degenerate."""
    base = SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** (layer_idx / (NUM_LAYERS - 1))
    head_factor = 0.7 + 0.6 * (head_idx / (NUM_HEADS - 1))  # 0.7 .. 1.3
    return float(base * head_factor)


def model_fn(input_ids: np.ndarray, layer_idx: int, head_idx: int) -> np.ndarray:
    """Return attention weights (1, seq_len, seq_len) for one head.

    The score matrix is Q @ K^T with HAND-SET projections. With
        q_i = [i / sigma^2, -1 / (2 sigma^2)]   (from f(i) via W_Q)
        k_j = [j, j^2]                            (from f(j) via _W_K)
    we get  q_i . k_j = i*j/sigma^2 - j^2/(2 sigma^2)
                      = -(j - i)^2 / (2 sigma^2) + i^2/(2 sigma^2),
    and the i^2 term is constant across keys, so softmax yields a Gaussian
    centred on the query with std sigma. Cross-chunk keys are masked out.
    """
    _, L = input_ids.shape
    if L != SEQ_LEN:
        raise ValueError(f"expected seq_len {SEQ_LEN}, got {L}")

    sigma = _sigma(layer_idx, head_idx)
    inv = 1.0 / (sigma * sigma)

    # Hand-set QUERY projection: f(i) = [i, i^2, 1] -> q_i = [i*inv, -0.5*inv].
    W_Q = torch.tensor(
        [[inv, 0.0],
         [0.0, 0.0],
         [0.0, -0.5 * inv]],
        dtype=torch.float32, device=DEVICE,
    )
    Q = _P @ W_Q                       # (256, 2)
    scores = Q @ _K.t()                # (256, 256) real QK on GPU
    scores = scores.masked_fill(~_same_chunk, float("-inf"))
    attn = torch.softmax(scores, dim=-1)          # rows sum to 1
    return attn.unsqueeze(0).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    payload = task.evaluate(model_fn)
    out_dir = results_dir(__file__)
    record_benchmark(__file__, out_dir, payload)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
