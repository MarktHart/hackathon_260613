"""Attention-boundary detection — pass_3 (hand_built, genuinely computed on GPU).

Approach (a small delta from experiments/base_model.py):
  base_model.py computes attention as softmax(Q @ K^T / sqrt(d)) with *learned*
  QKV projections. Here we keep the exact same dot-product-attention mechanism
  but HAND-SET a 2-D Q/K projection whose features are derived FROM THE TOKENS:

    feature 0  segment-sign  s_i = sign(pos_i - delim_pos)   in {-1, 0, +1}
                 (delim_pos is *detected* per sequence as argmax of tok==delim)
    feature 1  a special-token penalty that suppresses the delimiter and EOS

  Concretely, for head h with strength alpha_h:
      Q_i = [ sqrt(alpha_h) * s_i ,            1                 ]
      K_j = [ sqrt(alpha_h) * s_j ,  -LAMBDA * is_special_j      ]
      score_ij = Q_i . K_j = alpha_h * s_i*s_j  -  LAMBDA * is_special_j

  A content query on one side of the delimiter (s_i = +/-1) gets +alpha to keys
  on its own side, -alpha across the boundary, and -LAMBDA to delim/EOS, so the
  softmax concentrates *within its own segment*. This is real QK attention, not
  a pre-baked attention matrix.

Faithfulness check (computed here, shown in app): zeroing feature 0 (alpha=0)
ablates the segment mechanism -> attention falls back to uniform-over-content,
sharpness collapses to ~0 (== the linear baseline). The behaviour is causally
the segment-sign feature.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # the pipeline guarantees a visible GPU; no CPU fallback.

task = load_task(__file__)

# --- hand-set circuit hyper-parameters ------------------------------------
LAMBDA = 20.0                      # special-token (delim/EOS) suppression
HEAD_ALPHAS = [3.0, 5.0, 8.0, 12.0]  # one boundary head per attention head (4)

# Region layout (fixed by task.generate): segA[0:8] DELIM[8] segB[9:17] EOS[17]
SEG_LEN = 8
DELIM_POS = 8


# --------------------------------------------------------------------------
# The actual model: genuine dot-product attention on the GPU.
# --------------------------------------------------------------------------
def _features(input_ids: np.ndarray, delim_id: int):
    """Derive (segment_sign, is_special) features from the raw token ids."""
    tok = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=DEVICE)
    B, L = tok.shape
    eos_id = int(delim_id) - 1  # task: delim = vocab-1, eos = vocab-2

    pos = torch.arange(L, device=DEVICE, dtype=torch.float32)
    is_delim = (tok == int(delim_id)).float()        # (B, L)
    is_eos = (tok == eos_id).float()                 # (B, L)
    is_special = (is_delim + is_eos).clamp(max=1.0)  # (B, L)

    # Detect the delimiter position per sequence (exactly one delimiter).
    delim_pos = (is_delim * pos).sum(-1) / is_delim.sum(-1).clamp(min=1.0)  # (B,)
    seg_sign = torch.sign(pos[None, :] - delim_pos[:, None])  # (B, L) in {-1,0,1}
    return seg_sign, is_special


def attention(input_ids: np.ndarray, delim_id: int, alphas, lam: float) -> torch.Tensor:
    """Real QK attention. Returns (B, H, L, L) on the GPU, rows sum to 1."""
    seg_sign, is_special = _features(input_ids, delim_id)  # (B, L)
    B, L = seg_sign.shape
    H = len(alphas)
    out = torch.empty((B, H, L, L), device=DEVICE, dtype=torch.float32)
    ones = torch.ones_like(seg_sign)
    for h, alpha in enumerate(alphas):
        a = float(alpha) ** 0.5
        Q = torch.stack([a * seg_sign, ones], dim=-1)               # (B, L, 2)
        K = torch.stack([a * seg_sign, -lam * is_special], dim=-1)  # (B, L, 2)
        scores = torch.matmul(Q, K.transpose(1, 2))                 # (B, L, L)
        out[:, h] = torch.softmax(scores, dim=-1)
    return out


def model_fn(input_ids: np.ndarray, delim_id: int) -> np.ndarray:
    """task.evaluate contract: (B, seq_len) int + delim_id -> (B, H, L, L)."""
    attn = attention(input_ids, delim_id, HEAD_ALPHAS, LAMBDA)
    return attn.detach().cpu().numpy().astype(np.float32)


# --------------------------------------------------------------------------
# Analysis helpers (CPU/NumPy) used only to build the demo artefacts.
# --------------------------------------------------------------------------
def _seg_masses(attn: np.ndarray, q_slice, w0, w1, c0, c1):
    """Per-head region masses for one set of query positions. attn: (B,H,L,L)."""
    qa = attn[:, :, q_slice, :]                       # (B, H, q, L)
    within = qa[:, :, :, w0:w1].sum(-1)               # (B, H, q)
    delim = qa[:, :, :, DELIM_POS]
    cross = qa[:, :, :, c0:c1].sum(-1)
    eos = qa[:, :, :, -1]
    hw = within.mean(axis=(0, 2))
    hd = delim.mean(axis=(0, 2))
    hc = cross.mean(axis=(0, 2))
    he = eos.mean(axis=(0, 2))
    sharp = hw - np.maximum(np.maximum(hd, hc), he)   # (H,)
    return {
        "within": hw.tolist(), "delim": hd.tolist(),
        "cross": hc.tolist(), "eos": he.tolist(), "sharpness": sharp.tolist(),
    }


def per_head_masses(attn: np.ndarray):
    segA = _seg_masses(attn, slice(0, SEG_LEN),
                       0, SEG_LEN, DELIM_POS + 1, DELIM_POS + 1 + SEG_LEN)
    segB = _seg_masses(attn, slice(DELIM_POS + 1, DELIM_POS + 1 + SEG_LEN),
                       DELIM_POS + 1, DELIM_POS + 1 + SEG_LEN, 0, SEG_LEN)
    return {"segA": segA, "segB": segB}


def mean_sharpness(attn: np.ndarray) -> float:
    m = per_head_masses(attn)
    return float(np.mean(m["segA"]["sharpness"] + m["segB"]["sharpness"]))


# --------------------------------------------------------------------------
def main():
    batch = task.generate(seed=0)
    input_ids, delim_id = batch.input_ids, batch.delim_id

    # 1) Score on the canonical batch through the real model.
    payload = task.evaluate(model_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # 2) Demo artefacts -----------------------------------------------------
    attn = model_fn(input_ids, delim_id)                 # (B, 4, 18, 18)
    np.save(run_dir / "attn.npy", attn)
    np.save(run_dir / "input_ids.npy", input_ids)

    # Operating range: sweep alpha over 4 orders of magnitude (single head).
    alpha_sweep = []
    for alpha in np.logspace(-2, 2, 17):
        a_attn = attention(input_ids, delim_id, [float(alpha)], LAMBDA)
        a_attn = a_attn.detach().cpu().numpy().astype(np.float32)
        alpha_sweep.append({"alpha": float(alpha), "sharpness": mean_sharpness(a_attn)})

    # Faithfulness ablation: zero the segment-sign feature (alpha -> 0).
    abl = attention(input_ids, delim_id, [0.0], LAMBDA).detach().cpu().numpy().astype(np.float32)
    full = attention(input_ids, delim_id, [HEAD_ALPHAS[2]], LAMBDA).detach().cpu().numpy().astype(np.float32)

    meta = {
        "version": payload["version"],
        "config": payload["config"],
        "head_alphas": HEAD_ALPHAS,
        "lambda": LAMBDA,
        "per_head": per_head_masses(attn),
        "alpha_sweep": alpha_sweep,
        "ablation": {
            "full_sharpness": mean_sharpness(full),
            "ablated_sharpness": mean_sharpness(abl),
            "baseline_sharpness": 0.0,
        },
        "headline_sharpness": mean_sharpness(attn),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"headline boundary sharpness = {meta['headline_sharpness']:.4f}")
    print(f"ablation (segment feature off) = {meta['ablation']['ablated_sharpness']:.4f}")
    print(f"results -> {run_dir}")


if __name__ == "__main__":
    main()
