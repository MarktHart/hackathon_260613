"""
attention_histogram / first_pass — hand-built temperature-scaled attention head.

Mechanism (the delta from base_model.py):
    base_model.py scores keys against the query with a matched filter
    (q @ kᵀ) and divides by sqrt(head_dim) inside scaled_dot_product_attention.
    Here we keep the *same* matched-filter score — which is the optimal way to
    pick the target direction out of distractors when the query is a noisy copy
    of the target — but replace the fixed 1/sqrt(d) temperature with a single
    hand-set inverse-temperature β. Cosine-normalising q and k first makes β
    interpretable (scores live in [-1, 1]); β then controls how sharply the
    softmax concentrates on the matched key.

This is a hand_built attempt: no training, weights are set by hand. The only
free knob is β. Larger β ⇒ lower-entropy (sharper) histogram with the same
argmax, so it beats the plain dot-product baseline on sharpness while keeping
matched-filter targeting accuracy.

Runs the real compute on CUDA, as required by the framework.
"""

import argparse
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # framework guarantees a visible GPU; no CPU fallback.

task = load_task(__file__)


def make_model_fn(beta: float):
    """Return a model_fn: (query (d,), keys (n,d)) -> logits (n,), on GPU."""

    def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
        kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        # Cosine-normalise so β is a pure temperature on scores in [-1, 1].
        qn = qt / (qt.norm() + 1e-8)
        kn = kt / (kt.norm(dim=1, keepdim=True) + 1e-8)
        scores = kn @ qn                      # (n,) matched-filter cosine scores
        logits = beta * scores                # temperature-scaled (the mechanism)
        return logits.detach().cpu().numpy()

    return model_fn


def _softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=12.0,
                    help="inverse-temperature for the matched-filter scores")
    args = ap.parse_args()

    model_fn = make_model_fn(args.beta)

    # Headline payload via the canonical evaluator.
    payload = task.evaluate(model_fn)
    payload["model_name"] = f"temp_scaled_matched_filter(beta={args.beta:g})"

    run_dir = results_dir(__file__)

    # --- artefacts for the Demo tab: one representative histogram per sim ---
    batch = task.generate(seed=task.EVAL_SEED)
    n_pos = task.N_POSITIONS
    examples = []
    for ci, sim in enumerate(task.KEY_SIM_SWEEP):
        idx = ci * task.N_SEEDS  # first seed at this sweep point
        q, k = batch.queries[idx], batch.keys[idx]
        tgt = batch.target_index[idx]
        mech_attn = _softmax(np.asarray(model_fn(q, k), dtype=np.float64))
        base_attn = _softmax(k.astype(np.float64) @ q.astype(np.float64))
        examples.append({
            "similarity": float(sim),
            "target_index": int(tgt),
            "mech_attn": mech_attn.tolist(),
            "base_attn": base_attn.tolist(),
        })

    with open(run_dir / "payload.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(run_dir / "examples.json", "w") as f:
        json.dump({"n_positions": n_pos, "beta": args.beta,
                   "examples": examples}, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    # quick console summary
    print(f"beta={args.beta}  model={payload['model_name']}")
    for s, b in zip(payload["sweep"], payload["linear_baseline"]):
        print(f"  sim={s['similarity']:.1f}  sharp={s['attention_sharpness']:.3f} "
              f"(base {b['attention_sharpness']:.3f})  hit={s['target_hit_rate']:.3f} "
              f"(base {b['target_hit_rate']:.3f})")


if __name__ == "__main__":
    main()
