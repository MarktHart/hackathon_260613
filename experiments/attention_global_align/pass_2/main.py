"""attention_global_align / pass_2.

Hand-built retrieval head = `base_model.py` attention with ONE delta: the
dot-product score `K @ q` is multiplied by a scalar **temperature** `beta`
before the softmax (base_model scales by a fixed `1/sqrt(d)`; we make that
scale a settable parameter and turn it *up* so the head concentrates).

Why this is the fix: with raw logits `K @ q` the target logit is exactly 1
(target key == query, both unit) and the distractor logit is `cos`, but the
softmax over L=12 keys is far too soft, so the target only gets ~0.19 mass.
Multiplying by beta sharpens the distribution: the target (the unique
argmax for every cos < 1) captures almost all the mass. At cos == 1 the
distractor key is *mathematically identical* to the target, so the mass can
only ever split 50/50 there — that is the true alignment ceiling under
maximum interference, and it caps robustness at 0.5.

This module also computes explicit strawman curves (raw beta=1, a
temperature=0 ablation that collapses to uniform, and random logits) and
saves them next to the benchmark for the Demo tab.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback.


def make_tempered_fn(beta: float):
    """Return the hand-built head: logits = beta * (K @ q), computed on the GPU."""

    def model_fn(q: np.ndarray, K: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
        Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
        # The retrieval circuit: score every key by alignment with the query,
        # then a temperature controls how sharply attention concentrates.
        logits = beta * (Kt @ qt)
        return logits.detach().cpu().numpy()

    return model_fn


def _alignment_curve(task, model_fn) -> list[float]:
    """Run the canonical sweep with `model_fn`, return per-slice target mass."""
    payload = task.evaluate(model_fn)
    return [float(r["global_alignment"]) for r in payload["sweep"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=16.0,
                    help="temperature multiplier on K@q (default 16)")
    args = ap.parse_args()

    task = load_task(__file__)

    # --- The scored mechanism: tempered retrieval head on the GPU ---
    tempered_fn = make_tempered_fn(args.beta)
    payload = task.evaluate(tempered_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # --- Strawmen + ablation, evaluated under the identical condition ---
    # raw_fn  : beta = 1   -> the un-sharpened K@q (the first_pass mechanism)
    # zero_fn : beta = 0   -> temperature ablation; logits all 0 -> uniform head
    # random  : task's reference random-logit head (no query dependence at all)
    raw_curve = _alignment_curve(task, make_tempered_fn(1.0))
    zero_curve = _alignment_curve(task, make_tempered_fn(0.0))
    random_curve = _alignment_curve(task, task.random_model_fn())

    ours_curve = [float(r["global_alignment"]) for r in payload["sweep"]]
    dist_curve = [float(r["distractor_mass"]) for r in payload["sweep"]]
    uniform_curve = [float(r["global_alignment"]) for r in payload["uniform_baseline"]]
    cos_sweep = [float(c) for c in payload["distractor_cos_sweep"]]

    comparison = {
        "beta": float(args.beta),
        "canonical_cos": float(payload["canonical_distractor_cos"]),
        "seq_len": int(payload["seq_len"]),
        "distractor_cos_sweep": cos_sweep,
        "uniform_baseline": uniform_curve,
        "robustness_ceiling": 0.5,  # mass can only split 50/50 at cos==1
        "variants": {
            "tempered head (ours, beta=%g)" % args.beta: ours_curve,
            "raw K@q (beta=1)": raw_curve,
            "temperature=0 ablation": zero_curve,
            "random logits": random_curve,
        },
        "ours_distractor_mass": dist_curve,
    }
    with open(run_dir / "comparison.json", "w") as fh:
        json.dump(comparison, fh, indent=2)

    can_i = cos_sweep.index(payload["canonical_distractor_cos"])
    print(f"beta={args.beta}  canonical alignment={ours_curve[can_i]:.3f}  "
          f"raw={raw_curve[can_i]:.3f}  uniform={uniform_curve[can_i]:.3f}")
    print(f"robustness = align(cos=1)/align(cos=0) = "
          f"{ours_curve[-1]:.3f}/{ours_curve[0]:.3f} = "
          f"{ours_curve[-1] / max(ours_curve[0], 1e-9):.3f}")


if __name__ == "__main__":
    main()
