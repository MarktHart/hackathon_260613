"""attention_hierarchical_pool / pass_4 — hand_built.

A single attention-only head, NO mask, whose effective score matrix is a genuine
QK product of HAND-SET projections acting on positional features. The kernel is

    score_ij = -(i - j)^2 / (2 * sigma_L^2)

so softmax over keys is an (unmasked) Gaussian centred on the query with std
sigma_L. The ONLY quantity that changes with depth is sigma_L, which grows
geometrically across the 12 layers from a sub-token width (layer 0) to a
super-chunk width (layer 11).

WHY UNMASKED (the delta from pass_3): pass_3 hard-masked attention to the
query's own chunk, which made `superchunk_concentration` trivially 1.0 and never
demonstrated the chunk -> super-chunk pooling the goal asks about. Here the
Gaussian is free to spill across boundaries, so ALL three concentrations are
genuinely measured numbers. As depth rises the field sweeps through every level
of the tree:

    early  (small sigma): mass on the query token        -> high local
    mid    (sigma ~ 8):   mass fills the 16-tok chunk     -> chunk_conc high, local low
    late   (sigma ~ 16):  mass spreads across the 64-tok  -> chunk_conc DROPS to ~0.38
                          super-chunk                        while superchunk_conc
                                                             stays ~0.95

i.e. mass demonstrably LEAVES the chunk and pools at the super-chunk level — the
fine -> coarse hierarchical-pooling signature, every transition exhibited rather
than hard-coded.

Delta from base_model.py: one attention layer, no MLP, no residual; the QKV
projections are hand-set (not learned) and positional rather than from a token
embedding, and sigma is indexed by layer so the receptive field widens with
depth. Everything below runs as real Q@K^T in torch on CUDA. A faithfulness
ablation (flatten the sigma schedule -> robustness collapses to ~1.0) is RUN and
its artefact saved next to benchmark.json.
"""

import json
import os

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
SUPERCHUNK_SIZE = 4  # 4 chunks = 64-token super-chunk

# Receptive-field schedule. sigma grows geometrically from a sub-token local
# window (layer 0) up to a super-chunk-scale width (layer 11). SIGMA_MAX = 16 is
# deliberately ~ a quarter of the 64-token super-chunk: wide enough that late
# layers spill out of the 16-token chunk and fill the super-chunk, but not so
# wide that mass escapes the super-chunk (so superchunk_conc stays high, ~0.95).
SIGMA_MIN = 0.60   # layer 0  -> essentially attends to the query token (fine)
SIGMA_MAX = 16.0   # layer 11 -> spreads across the 64-token super-chunk (coarse)

# Positional feature matrix f(p) = [p, p^2, 1] (the fixed "base-model" positional
# encoding). The whole head-specific circuit lives in the hand-set projections.
# float64 to keep the QK product numerically clean for the tight early-layer
# Gaussians (avoids catastrophic-cancellation noise near the diagonal).
_pos = torch.arange(SEQ_LEN, dtype=torch.float64, device=DEVICE)
_P = torch.stack([_pos, _pos * _pos, torch.ones_like(_pos)], dim=1)  # (256, 3)

# Hand-set KEY projection: f(j) = [j, j^2, 1] -> k_j = [j, j^2].
_W_K = torch.tensor(
    [[1.0, 0.0],
     [0.0, 1.0],
     [0.0, 0.0]],
    dtype=torch.float64, device=DEVICE,
)
_K = _P @ _W_K  # (256, 2), rows = [j, j^2]


def _sigma(layer_idx: int, head_idx: int) -> float:
    """Per-(layer, head) Gaussian width. Base widens geometrically with depth;
    heads get a mild spread so a layer is not degenerate."""
    base = SIGMA_MIN * (SIGMA_MAX / SIGMA_MIN) ** (layer_idx / (NUM_LAYERS - 1))
    head_factor = 0.75 + 0.5 * (head_idx / (NUM_HEADS - 1))  # 0.75 .. 1.25
    return float(base * head_factor)


def _attn_for_sigma(sigma: float) -> np.ndarray:
    """Real Q@K^T attention for a given Gaussian width, on the GPU. No mask.

    With q_i = [i/sigma^2, -1/(2 sigma^2)] and k_j = [j, j^2]:
        q_i . k_j = i*j/sigma^2 - j^2/(2 sigma^2)
                  = -(i - j)^2 / (2 sigma^2) + i^2/(2 sigma^2),
    and the i^2 term is a per-query constant removed by softmax, so the row is a
    Gaussian over keys centred on the query with std sigma."""
    inv = 1.0 / (sigma * sigma)
    W_Q = torch.tensor(
        [[inv, 0.0],
         [0.0, 0.0],
         [0.0, -0.5 * inv]],
        dtype=torch.float64, device=DEVICE,
    )
    Q = _P @ W_Q                       # (256, 2)
    scores = Q @ _K.t()                # (256, 256) genuine QK on GPU, unmasked
    attn = torch.softmax(scores, dim=-1)
    return attn.unsqueeze(0).to(torch.float32).detach().cpu().numpy()


def model_fn(input_ids: np.ndarray, layer_idx: int, head_idx: int) -> np.ndarray:
    """Canonical model: width set by the per-depth schedule."""
    _, L = input_ids.shape
    if L != SEQ_LEN:
        raise ValueError(f"expected seq_len {SEQ_LEN}, got {L}")
    return _attn_for_sigma(_sigma(layer_idx, head_idx))


def flat_model_fn(input_ids: np.ndarray, layer_idx: int, head_idx: int) -> np.ndarray:
    """Faithfulness ablation: SAME kernel, but sigma is frozen at the geometric
    mean width for every layer (depth dependence removed). The fine->coarse shift
    is the only causal knob, so this should collapse the headline to ~1.0."""
    _, L = input_ids.shape
    if L != SEQ_LEN:
        raise ValueError(f"expected seq_len {SEQ_LEN}, got {L}")
    sigma_const = float(np.sqrt(SIGMA_MIN * SIGMA_MAX))  # geometric mean
    head_factor = 0.75 + 0.5 * (head_idx / (NUM_HEADS - 1))
    return _attn_for_sigma(sigma_const * head_factor)


def _robustness_and_spreads(payload: dict) -> tuple[float, list[float]]:
    """Replicate benchmark.py's headline: spread_L = chunk/local, ratio of
    median(late spread) to median(early spread)."""
    def med(xs: list[float]) -> float:
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    by_layer: dict[int, list[dict]] = {L: [] for L in range(NUM_LAYERS)}
    for rec in payload["sweep"]:
        by_layer[rec["layer"]].append(rec)
    spreads: list[float] = []
    for L in range(NUM_LAYERS):
        recs = by_layer[L]
        loc = med([r["local_concentration"] for r in recs])
        ch = med([r["chunk_concentration"] for r in recs])
        spreads.append(ch / loc if loc > 0 else float("nan"))
    early = med(spreads[: NUM_LAYERS // 2])
    late = med(spreads[NUM_LAYERS // 2:])
    return (late / early if early > 0 else float("nan")), spreads


def main() -> None:
    out_dir = results_dir(__file__)

    # 1) Canonical run -> benchmark.json
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, out_dir, payload)
    canon_rob, canon_spreads = _robustness_and_spreads(payload)

    # 2) Faithfulness ablation (RUN, not argued): flatten sigma -> shift dies.
    flat_payload = task.evaluate(flat_model_fn)
    flat_rob, flat_spreads = _robustness_and_spreads(flat_payload)

    ablation = {
        "description": (
            "Knock out the only causal knob (the depth-indexed Gaussian width) "
            "by freezing sigma at its geometric mean for every layer. The "
            "fine->coarse shift vanishes and the headline collapses to ~1.0, "
            "confirming the schedule — not anything else — produces the hierarchy."
        ),
        "canonical_robustness": canon_rob,
        "flat_sigma_robustness": flat_rob,
        "uniform_baseline_robustness": 1.0,
        "canonical_spread_per_layer": canon_spreads,
        "flat_spread_per_layer": flat_spreads,
        "sigma_min": SIGMA_MIN,
        "sigma_max": SIGMA_MAX,
    }
    with open(os.path.join(out_dir, "ablation.json"), "w", encoding="utf-8") as f:
        json.dump(ablation, f, indent=2)

    print(f"Results written to {out_dir}")
    print(f"canonical robustness = {canon_rob:.3f}  |  flat-sigma ablation = {flat_rob:.3f}")


if __name__ == "__main__":
    main()
