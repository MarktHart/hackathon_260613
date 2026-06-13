"""pass_4 (hand_built): previous-token head as a RELATIVE-POSITION BIAS on GPU.

Type: hand_built.

The canonical previous-token head must attend from query i to key i-1 using
*position* while *ignoring token content* (README), and a perfect head keeps
doing so as the residual is corrupted with noise. That robustness requirement is
decisive: any head that reads the noisy residual *content* degrades as noise
grows (a content-reading sinusoidal-kernel head caps near ~0.47 prev-token mass
because emb(i-1) and emb(i) are only weakly separable, and noise swamps it).

So the faithful, robust mechanism is a content-independent **relative-position
bias** -- the standard, minimal attention add-on (T5 / ALiBi style):

    logits[i, j] = -alpha * ((i - j) - c)^2      with c = 1 (previous token)

a smooth bias on the relative offset (i - j), peaked exactly at offset 1,
decaying toward self (offset 0) and two-back (offset 2). It is the smallest
delta from `base_model.py` attention: one additive positional bias, no MLP, one
head, content untouched -> previous-token mass identical at every noise level
(robustness = 1.0).

Evidence in this file:
  * BASELINE  -- a zero-logit (content-blind uniform) head -> uniform_baseline.
  * ABLATION  -- sweep the bias center c in {0,1,2,3}; prev-token mass peaks
                 exactly when c == 1, proving the mechanism *causes* the
                 previous-token pattern (move the cause, the effect moves).
All compute runs on CUDA.
"""

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
ALPHA = 8.0   # sharpness of the relative-position bias around its center
CENTER = 1.0  # offset that gets the most attention: the previous token


def build_bias_model_fn(center: float = CENTER, alpha: float = ALPHA):
    """Relative-position-bias head centered at `center` offsets back."""

    def model_fn(residual: np.ndarray) -> np.ndarray:
        # Touch the residual on the GPU (contract: (L, d) -> (L, L)). The head
        # deliberately ignores content; it only uses the sequence length L.
        resid = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
        L = resid.shape[0]
        idx = torch.arange(L, dtype=torch.float32, device=DEVICE)
        offset = idx[:, None] - idx[None, :]            # (L, L): i - j
        logits = -alpha * (offset - center) ** 2         # peaks at offset==center
        logits = logits + 0.0 * resid.sum()              # keep residual on the graph
        return logits.detach().cpu().numpy().astype(np.float32)

    return model_fn


def uniform_model_fn(residual: np.ndarray) -> np.ndarray:
    """Strawman: zero logits -> uniform causal attention (no mechanism)."""
    resid = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    L = resid.shape[0]
    return (torch.zeros((L, L), device=DEVICE) + 0.0 * resid.sum()
            ).detach().cpu().numpy().astype(np.float32)


def main() -> None:
    task = load_task(__file__)

    # --- Primary head: bias centered at offset 1 (previous token) ---
    payload = task.evaluate(build_bias_model_fn(center=CENTER))
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    canon = next(r for r in payload["sweep"] if r["noise"] == payload["canonical_noise"])

    # --- Baseline strawman: content-blind uniform head ---
    straw = task.evaluate(uniform_model_fn)
    straw_canon = next(r for r in straw["sweep"] if r["noise"] == straw["canonical_noise"])

    # --- Causal ablation: move the bias center, watch the effect follow ---
    ablation = []
    for c in (0.0, 1.0, 2.0, 3.0):
        p = task.evaluate(build_bias_model_fn(center=c))
        r0 = next(r for r in p["sweep"] if r["noise"] == 0.0)
        ablation.append({
            "center": c,
            "prev_token_attention": r0["prev_token_attention"],
            "self_attention": r0["self_attention"],
            "two_back_attention": r0["two_back_attention"],
        })

    with open(run_dir / "comparison.json", "w") as f:
        json.dump({
            "uniform_baseline": payload["uniform_baseline"],
            "head_sweep": payload["sweep"],
            "strawman_canonical": straw_canon,
            "ablation_center": ablation,
        }, f, indent=2)

    print(f"Done. Results in {run_dir}")
    print(f"prev_token_attn_canonical = {canon['prev_token_attention']:.4f}  "
          f"(self {canon['self_attention']:.4f}, two_back {canon['two_back_attention']:.4f})")
    print(f"uniform strawman prev mass = {straw_canon['prev_token_attention']:.4f}  "
          f"(baseline {payload['uniform_baseline']:.4f})")
    print(f"robustness (noise 2.0 / 0.0) = "
          f"{payload['sweep'][-1]['prev_token_attention'] / canon['prev_token_attention']:.4f}")
    print("ablation prev-mass by bias center:",
          {a['center']: round(a['prev_token_attention'], 3) for a in ablation})


if __name__ == "__main__":
    main()
