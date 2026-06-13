"""
attention_histogram / pass_2 — hand-built ITERATIVE-REFINEMENT attention head.

Delta from base_model.py / from pass_1:
    base_model.py scores keys against the query once (q @ kᵀ inside
    scaled_dot_product_attention). pass_1 kept that single matched filter and
    only added a temperature β — so it sharpened the histogram but its argmax
    (targeting) was IDENTICAL to plain dot-product. Here the mechanism instead
    DENOISES the query with the keys before scoring:

        a0  = softmax(β0 · K q)            # first, soft attention with noisy q
        q1  = unit(Kᵀ a0)                  # refined query = key-weighted average
        ... repeat n_iter times ...
        out = β · K q_final                # final logits

    The refined query is a convex combination of keys dominated by the target,
    so it is a *better* estimate of the target direction t than the
    noise-corrupted query q. This is one step of the attention power-iteration
    that base_model.py's residual stream would implement across layers. It
    improves BOTH histogram sharpness AND target hit-rate, unlike a pure
    temperature change.

Hand_built attempt: no training. Knobs are hand-set (β0, β, n_iter).
Causal ablation (n_iter=0) is the no-refinement control = the plain matched
filter; main.py evaluates it explicitly so the refinement's contribution is
measured, not assumed. Real compute runs in torch on CUDA.
"""

import argparse
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # framework guarantees a visible GPU; no CPU fallback.

task = load_task(__file__)


def make_model_fn(beta0: float, beta: float, n_iter: int):
    """Return model_fn: (query (d,), keys (n,d)) -> logits (n,), on GPU.

    n_iter=0 disables refinement -> plain matched filter (the ablation).
    """

    def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        qt = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
        kt = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        qn = qt / (qt.norm() + 1e-8)
        kn = kt / (kt.norm(dim=1, keepdim=True) + 1e-8)  # (n,d) unit keys

        q_cur = qn
        for _ in range(n_iter):
            a = torch.softmax(beta0 * (kn @ q_cur), dim=0)   # (n,) attention
            q_ref = kn.t() @ a                               # (d,) key-avg
            q_cur = q_ref / (q_ref.norm() + 1e-8)            # renormalise
        logits = beta * (kn @ q_cur)                         # (n,) final logits
        return logits.detach().cpu().numpy()

    return model_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta0", type=float, default=8.0,
                    help="inverse-temp for the refinement attention step")
    ap.add_argument("--beta", type=float, default=16.0,
                    help="inverse-temp for the final scoring step")
    ap.add_argument("--n_iter", type=int, default=2,
                    help="number of query-refinement iterations")
    args = ap.parse_args()

    mech_fn = make_model_fn(args.beta0, args.beta, args.n_iter)
    ablate_fn = make_model_fn(args.beta0, args.beta, 0)  # no-refinement control

    payload = task.evaluate(mech_fn)
    payload["model_name"] = (
        f"iter_refine(beta0={args.beta0:g},beta={args.beta:g},"
        f"n_iter={args.n_iter})")

    # Ablation: same head with refinement removed (n_iter=0).
    ablate_payload = task.evaluate(ablate_fn)

    run_dir = results_dir(__file__)

    batch = task.generate(seed=task.EVAL_SEED)
    n_pos = task.N_POSITIONS

    def softmax_np(x):
        z = x - np.max(x)
        e = np.exp(z)
        return e / e.sum()

    examples = []
    for ci, sim in enumerate(task.KEY_SIM_SWEEP):
        idx = ci * task.N_SEEDS
        q, k = batch.queries[idx], batch.keys[idx]
        tgt = batch.target_index[idx]
        examples.append({
            "similarity": float(sim),
            "target_index": int(tgt),
            "mech_attn": softmax_np(
                np.asarray(mech_fn(q, k), np.float64)).tolist(),
            "ablate_attn": softmax_np(
                np.asarray(ablate_fn(q, k), np.float64)).tolist(),
            "base_attn": softmax_np(
                k.astype(np.float64) @ q.astype(np.float64)).tolist(),
        })

    with open(run_dir / "payload.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(run_dir / "ablation.json", "w") as f:
        json.dump(ablate_payload, f, indent=2)
    with open(run_dir / "examples.json", "w") as f:
        json.dump({"n_positions": n_pos, "examples": examples}, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    print(f"model={payload['model_name']}")
    for s, a, b in zip(payload["sweep"], ablate_payload["sweep"],
                       payload["linear_baseline"]):
        print(f"  sim={s['similarity']:.1f} "
              f"hit mech={s['target_hit_rate']:.2f} "
              f"no-refine={a['target_hit_rate']:.2f} "
              f"base={b['target_hit_rate']:.2f} | "
              f"sharp mech={s['attention_sharpness']:.2f} "
              f"base={b['attention_sharpness']:.2f}")


if __name__ == "__main__":
    main()
