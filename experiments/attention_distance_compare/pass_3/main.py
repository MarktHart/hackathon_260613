"""
attention_distance_compare / pass_3  —  hand-built RELATIVE-POSITION-BIAS heads.

Approach (hand_built, faithful mechanism): instead of writing the softmax
*output* directly (what pass_2 did), this attempt builds a real multi-head
self-attention forward pass — token embedding -> per-head Q/K projections ->
scaled dot-product scores -> softmax — and adds the SINGLE delta from
`base_model.py` that produces a distance preference: an additive ALiBi-style
relative-position bias  b_{l,h}(i,j) = -|i-j| / lambda_{l,h}  injected into the
attention logits before the softmax.

So the distance decay is *emitted by an actual attention computation* (content
QK noise included), not pasted into the result. The per-head decay strength is
controlled by lambda_{l,h}: small lambda -> steep local head, large lambda ->
flat global head. We deliberately make heads heterogeneous (local in head 0,
global in head 7; shallower layers more local) so the per-layer/head metrics
show the local-vs-global structure the goal asks about.

Faithfulness / causal check: `model_fn_ablated` runs the IDENTICAL forward pass
with the relative-position bias term zeroed. Knocking out that one circuit
collapses the decay back to the content-only (≈uniform) baseline — evidence the
bias term is what causes the distance structure. Both curves are saved for the
Demo tab.

Everything runs in torch on CUDA.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback.

# --- Canonical measurement condition (fixed by the goal) ---
N_LAYERS = 4
N_HEADS = 8
D_MODEL = 32
D_HEAD = 8
VOCAB_SIZE = 1000

# Per-head decay length-scales (small => local/steep, large => global/flat).
LAMBDAS_HEAD = [1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 20.0, 64.0]
# Layer modulation: shallower layers more local, deeper more global.
LAYER_FACTOR = [1.4, 1.0, 0.7, 0.45]
CONTENT_SCALE = 0.15  # weight on the (noisy) content QK term

DISTANCE_BIN_CENTERS = [0.5, 1.5, 2.5, 3.5, 4.5, 6.0, 9.0, 13.5, 24.5, 48.0]


# ----------------------------------------------------------------------------
# Hand-set weights (deterministic), built once on CUDA and reused.
# ----------------------------------------------------------------------------
def _build_weights():
    g = torch.Generator(device=DEVICE).manual_seed(0)
    embed = torch.randn(VOCAB_SIZE, D_MODEL, generator=g, device=DEVICE)
    # Per (layer, head) Q/K projection matrices: (L, H, D_MODEL, D_HEAD).
    wq = torch.randn(N_LAYERS, N_HEADS, D_MODEL, D_HEAD, generator=g, device=DEVICE)
    wk = torch.randn(N_LAYERS, N_HEADS, D_MODEL, D_HEAD, generator=g, device=DEVICE)
    # Relative-position bias slope per (layer, head): -1 / lambda.
    inv_lambda = torch.tensor(
        [[LAYER_FACTOR[l] / LAMBDAS_HEAD[h] for h in range(N_HEADS)]
         for l in range(N_LAYERS)],
        device=DEVICE, dtype=torch.float32,
    )  # (L, H)
    return embed, wq, wk, inv_lambda


_EMBED, _WQ, _WK, _INV_LAMBDA = _build_weights()


def _forward(input_ids: np.ndarray, use_bias: bool) -> np.ndarray:
    """Real multi-head attention forward; returns (L, H, B, S, S) softmax."""
    ids = torch.as_tensor(np.asarray(input_ids), dtype=torch.long, device=DEVICE)
    B, S = ids.shape
    x = _EMBED[ids]  # (B, S, D_MODEL)

    pos = torch.arange(S, device=DEVICE, dtype=torch.float32)
    dist = (pos[:, None] - pos[None, :]).abs()  # (S, S) = |i - j|

    out = torch.empty(N_LAYERS, N_HEADS, B, S, S, device=DEVICE)
    scale = 1.0 / math.sqrt(D_HEAD)
    for l in range(N_LAYERS):
        for h in range(N_HEADS):
            q = x @ _WQ[l, h]           # (B, S, D_HEAD)
            k = x @ _WK[l, h]           # (B, S, D_HEAD)
            scores = CONTENT_SCALE * (q @ k.transpose(-1, -2)) * scale  # (B,S,S)
            if use_bias:
                scores = scores - _INV_LAMBDA[l, h] * dist  # ALiBi-style delta
            attn = torch.softmax(scores, dim=-1)
            out[l, h] = attn
    return out.detach().cpu().numpy()


def model_fn(input_ids: np.ndarray) -> dict:
    """Distance-biased attention (the real mechanism)."""
    return {"attention": _forward(input_ids, use_bias=True)}


def model_fn_ablated(input_ids: np.ndarray) -> dict:
    """Same forward, relative-position bias zeroed (content-only control)."""
    return {"attention": _forward(input_ids, use_bias=False)}


# ----------------------------------------------------------------------------
# Decay-slope helper (mirrors benchmark.py so the Demo can label heads).
# ----------------------------------------------------------------------------
def _decay_slope(bins, vals) -> float:
    xs, ys = [], []
    for d, v in zip(bins, vals):
        if d < 1:
            continue
        xs.append(math.log2(float(d)))
        ys.append(math.log(max(float(v), 1e-12)))
    if len(xs) < 2:
        return 0.0
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((xx - mx) ** 2 for xx in xs)
    if denom <= 1e-12:
        return 0.0
    slope = sum((xx - mx) * (yy - my) for xx, yy in zip(xs, ys)) / denom
    return float(-slope)


def main() -> None:
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # Causal/faithfulness control: same model with the bias circuit removed.
    ablated_payload = task.evaluate(model_fn_ablated)

    bins = payload["distance_bins"]
    per_lh = payload["mean_attn_per_layer_head_bin"]  # (L, H, 10)
    lh_slopes = [[_decay_slope(bins, per_lh[l][h]) for h in range(N_HEADS)]
                 for l in range(N_LAYERS)]

    demo = {
        "distance_bins": bins,
        "mean_attn_per_bin": payload["mean_attn_per_bin"],
        "uniform_baseline_per_bin": payload["uniform_baseline_per_bin"],
        "ablated_mean_attn_per_bin": ablated_payload["mean_attn_per_bin"],
        "mean_attn_per_layer_head_bin": per_lh,
        "layer_head_slope": lh_slopes,
        "headline_slope": _decay_slope(bins, payload["mean_attn_per_bin"]),
        "ablated_slope": _decay_slope(bins, ablated_payload["mean_attn_per_bin"]),
        "lambdas_head": LAMBDAS_HEAD,
        "layer_factor": LAYER_FACTOR,
    }
    with open(run_dir / "demo.json", "w") as f:
        json.dump(demo, f, indent=2)

    print(f"headline decay slope (model)   = {demo['headline_slope']:.3f}")
    print(f"decay slope (bias ablated)     = {demo['ablated_slope']:.3f}")
    print(f"saved artefacts to {run_dir}")


if __name__ == "__main__":
    main()
