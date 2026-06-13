"""
Hand-built hierarchical-pooling attention.

Contract (task.py, current):
    model_fn(input_ids: (batch, seq_len) int32, layer_idx: int, head_idx: int)
        -> (batch, seq_len, seq_len) float32, rows sum to 1.

Hypothesis: attention pools fine -> coarse with depth. We hand-build heads so
that EARLY layers attend in a tight local window (±1-2 tokens, sharp, low
entropy => spread = chunk_conc/local_conc close to 1) and LATE layers attend
broadly across the query's own 16-token chunk (uniform-within-chunk =>
chunk_conc stays ~1 while local_conc drops, so spread grows). The benchmark's
headline `hierarchical_robustness_canonical = median(spread_late)/median(
spread_early)` is therefore > 1.

All compute runs on CUDA via torch.
"""
from __future__ import annotations

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by pipeline; no CPU fallback

task = load_task(__file__)

SEQ_LEN = 256
NUM_LAYERS = 12
NUM_HEADS = 8
CHUNK_SIZE = 16


def model_fn(input_ids: np.ndarray, layer_idx: int, head_idx: int) -> np.ndarray:
    """Return (batch, seq_len, seq_len) attention for one (layer, head)."""
    ids = torch.as_tensor(input_ids, dtype=torch.int64, device=DEVICE)  # (B, L)
    B, L = ids.shape

    pos = torch.arange(L, device=DEVICE, dtype=torch.float32)
    qpos = pos.view(L, 1)   # (L, 1) query position
    kpos = pos.view(1, L)   # (1, L) key position

    chunk = (pos // CHUNK_SIZE).to(torch.int64)        # (L,) chunk index per pos
    same_chunk = (chunk.view(L, 1) == chunk.view(1, L))  # (L, L) bool

    # Bandwidth grows with depth: early layers tight (local), late layers wide
    # (whole chunk). Add a small per-head jitter so heads aren't identical but
    # the layer trend dominates.
    depth = layer_idx / max(NUM_LAYERS - 1, 1)          # 0 .. 1
    head_jitter = 1.0 + 0.15 * (head_idx / max(NUM_HEADS - 1, 1))
    # sigma in tokens: ~0.6 (very local) early -> ~12 (covers the 16-tok chunk) late
    sigma = (0.6 + depth * 11.4) * head_jitter

    dist2 = (qpos - kpos) ** 2                            # (L, L)
    logits = -dist2 / (2.0 * sigma * sigma)               # (L, L) Gaussian over distance

    # Restrict attention to the query's own chunk so chunk_concentration ~ 1.0
    # (mass stays inside the chunk) while local_concentration falls with width.
    neg_inf = torch.finfo(torch.float32).min
    logits = torch.where(same_chunk, logits, torch.full_like(logits, neg_inf))

    attn = torch.softmax(logits, dim=1)                  # (L, L) rows sum to 1
    attn = attn.unsqueeze(0).expand(B, L, L).contiguous()  # (B, L, L)

    return attn.detach().cpu().numpy().astype(np.float32)


def main():
    payload = task.evaluate(model_fn)
    print("Got payload with keys:", sorted(payload.keys()))

    out_dir = results_dir(__file__)
    record_benchmark(__file__, out_dir, payload)
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
