"""attention_polyeval — pass_2 (hand_built)

Hypothesis: a SINGLE attention block can evaluate the quadratic x^2 elementwise.
The squaring nonlinearity is the bilinear QK product (Q = K = sqrt(beta)*x, so the
self-attention score is beta * x^2), and softmax over {self, constant-sink} keys
turns that score into a bounded weight that an affine output projection reads back
out as ~x^2.

This is `base_model.py`'s attention block specialised:
  - identity Q/K/V projections (each token attends to itself + one learned sink key)
  - the self-score is the diagonal QK quadratic form  beta * x_f^2  (per feature f)
  - a 2-key softmax (self vs. constant sink) is the only nonlinearity
  - W_O is an affine readout (alpha, gamma) calibrated to the known input range
No MLP, no second layer.

All forward passes run on CUDA (the QK product + softmax). Calibration of the
affine readout (alpha, gamma) is offline on a fixed-seed synthetic U[-s, s] sample
— it never looks at the evaluation targets; it only derives the W_O weights.
"""

import os
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback

task = load_task(__file__)

# ---- fixed circuit hyper-parameters (the "hand-set weights") --------------
BETA = 1.0   # QK gain: self-score = BETA * x^2
B0 = 0.0     # constant logit of the learned attention "sink" key


# ---------------------------------------------------------------------------
# Offline calibration of the affine output projection (alpha * p + gamma).
# Uses only the INPUT distribution U[-scale, scale]; never the eval targets.
# ---------------------------------------------------------------------------
def calibrate(beta: float, b0: float, scale: float, mode: str,
              n: int = 200_000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-scale, scale, size=n)
    u = x * x  # the quadratic target we want the readout to express
    if mode == "quad":
        ls = beta * x * x          # genuine QK quadratic score
    elif mode == "linear":
        ls = beta * x              # ablation: linear (non-quadratic) score
    else:
        raise ValueError(mode)
    p = 1.0 / (1.0 + np.exp(-(ls - b0)))   # softmax weight on the self key
    A = np.stack([np.ones_like(p), p], axis=1)
    coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    gamma, alpha = float(coef[0]), float(coef[1])
    return alpha, gamma


# ---------------------------------------------------------------------------
# GPU forward pass of the attention block.
# ---------------------------------------------------------------------------
def attention_forward(x_np: np.ndarray, beta: float, b0: float,
                      alpha: float, gamma: float, mode: str = "quad") -> np.ndarray:
    x = torch.as_tensor(x_np, dtype=torch.float32, device=DEVICE)
    if mode == "quad":
        self_logit = beta * x * x          # diagonal QK quadratic form  beta * x_f^2
    elif mode == "linear":
        self_logit = beta * x              # ablation: linear self-score
    else:
        raise ValueError(mode)
    sink_logit = torch.full_like(self_logit, float(b0))     # constant sink key
    logits = torch.stack([self_logit, sink_logit], dim=-1)  # [..., 2]
    p = torch.softmax(logits, dim=-1)[..., 0]               # weight on self key
    out = gamma + alpha * p                                 # affine W_O readout
    return out.detach().cpu().numpy().astype(np.float32)


def r2_vs(out: np.ndarray, target: np.ndarray) -> float:
    mse = float(np.mean((out - target) ** 2))
    var = float(np.var(target))
    return 1.0 - mse / var if var > 0 else 0.0


def main():
    run_dir = results_dir(__file__)

    # --- calibrate the canonical (scale=1.0) readout ----------------------
    alpha, gamma = calibrate(BETA, B0, scale=1.0, mode="quad")

    def model_fn(inputs: np.ndarray) -> np.ndarray:
        return attention_forward(inputs, BETA, B0, alpha, gamma, mode="quad")

    # --- canonical benchmark ----------------------------------------------
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    # --- faithfulness ablations at degree 2 (x^2) -------------------------
    batch = task.generate(seed=42)
    x = batch.inputs
    target2 = batch.targets[2]

    out_quad = attention_forward(x, BETA, B0, alpha, gamma, mode="quad")
    r2_quad = r2_vs(out_quad, target2)

    # Ablation A: replace the quadratic QK score with a LINEAR score.
    a_lin, g_lin = calibrate(BETA, B0, scale=1.0, mode="linear")
    out_lin = attention_forward(x, BETA, B0, a_lin, g_lin, mode="linear")
    r2_lin = r2_vs(out_lin, target2)

    # Linear baseline (best affine a*x+b) — taken straight from the payload.
    base2 = next(r for r in payload["linear_baseline"] if r["degree"] == 2)
    r2_base = float(base2["variance_explained"])

    ablation = {
        "degree": 2,
        "mechanism_quad_qk_r2": r2_quad,
        "ablation_linear_qk_r2": r2_lin,      # squaring removed -> collapses
        "linear_baseline_r2": r2_base,        # best affine map -> ~0
        "note": "Removing the quadratic QK self-score (linear-QK ablation) "
                "collapses R^2 to the linear-baseline floor; the softmax-over-"
                "QK-square is causally responsible for the fit.",
    }
    with open(os.path.join(run_dir, "ablation.json"), "w") as f:
        json.dump(ablation, f, indent=2)

    # --- operating range: >= 2 orders of magnitude of input scale ---------
    scale_sweep = []
    for scale in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        rng = np.random.default_rng(123)
        xs = rng.uniform(-scale, scale, size=(128, 64)).astype(np.float32)
        ts = (xs ** 2).astype(np.float32)
        # scale-adaptive QK gain keeps beta*x^2 in the informative range
        beta_s = 1.0 / (scale * scale)
        a_s, g_s = calibrate(beta_s, B0, scale=scale, mode="quad")
        out_s = attention_forward(xs, beta_s, B0, a_s, g_s, mode="quad")
        r2_s = r2_vs(out_s, ts)
        # fixed-beta (non-adaptive) variant, to show where it degrades
        a_f, g_f = calibrate(BETA, B0, scale=scale, mode="quad")
        out_f = attention_forward(xs, BETA, B0, a_f, g_f, mode="quad")
        r2_f = r2_vs(out_f, ts)
        scale_sweep.append({
            "scale": float(scale),
            "r2_adaptive_beta": float(r2_s),
            "r2_fixed_beta": float(r2_f),
        })
    with open(os.path.join(run_dir, "scale_sweep.json"), "w") as f:
        json.dump(scale_sweep, f, indent=2)

    # --- scatter sample for the Demo viz ----------------------------------
    flat_x = x.flatten()
    flat_out = out_quad.flatten()
    idx = np.argsort(flat_x)
    sel = idx[:: max(1, len(idx) // 2000)]  # even subsample for a clean curve
    scatter = {
        "x": flat_x[sel].astype(float).tolist(),
        "out": flat_out[sel].astype(float).tolist(),
        "beta": BETA,
        "b0": B0,
        "alpha": float(alpha),
        "gamma": float(gamma),
    }
    with open(os.path.join(run_dir, "scatter.json"), "w") as f:
        json.dump(scatter, f, indent=2)

    print(f"[pass_2] degree-2 R^2: mechanism={r2_quad:.4f} "
          f"linear-QK-ablation={r2_lin:.4f} linear-baseline={r2_base:.4f}")
    print(f"[pass_2] headline (poly_eval) = {r2_quad - r2_base:.4f}")
    print(f"[pass_2] artefacts -> {run_dir}")


if __name__ == "__main__":
    main()
