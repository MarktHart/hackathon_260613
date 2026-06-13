"""
attention_tsp / pass_3 — QK dot-product attention that *is* nearest-neighbour.

Approach (hand_built + interp ablation):

We take `base_model.py`'s single-head `Attention` and replace two pieces with
hand-set, fixed weights:

  1. A fixed *quadratic coordinate embedding* phi(x, y) = [x, y, x^2+y^2, 1]
     in place of the learned token-embedding table. Cities are continuous, so
     the "token embedding" is a feature map. This is the only nonlinearity.
  2. Hand-set Q/K projections (4x4) so the ordinary dot-product attention score
     between the current city i and any city j equals the NEGATIVE squared
     Euclidean distance:

         Q_i = phi_i = [x_i, y_i, s_i, 1]                      (W_q = diag(1,1,1,1))
         K_j = W_k phi_j = [2 x_j, 2 y_j, -1, -s_j]            (s = x^2+y^2)
         Q_i . K_j = 2(x_i x_j + y_i y_j) - s_i - s_j = -||c_i - c_j||^2

  argmax_j over unvisited of (Q_i . K_j) is therefore the nearest unvisited city.

So the NN heuristic is not bolted on with a custom distance call — it falls
straight out of a standard transformer attention score once the key carries a
-||k||^2 feature. That is the mechanistic claim, and we test it causally with
two ablations (run in main.py and saved for the Demo tab):

  * ablate_key_norm : zero the key's -s_j feature  -> score = 2 q.k_coord - s_i.
        The -s_i term is constant across j, so argmax picks the city with the
        LARGEST coordinate dot-product (farthest in the current direction), not
        the nearest. Accuracy collapses. This feature is NECESSARY.
  * ablate_query_norm : zero the query's s_i feature -> score shifts by +s_i,
        a constant across j. argmax is UNCHANGED. This feature is causally
        INERT — a control that pins down exactly which feature does the work.

Everything runs in torch on CUDA.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback.

# --- Hand-set attention weights (base_model.py Attention, frozen) ------------
# phi(x, y) = [x, y, s, 1] with s = x^2 + y^2.
# W_q maps phi -> Q ;  W_k maps phi -> K.
_W_Q_FULL = torch.tensor(
    [[1.0, 0.0, 0.0, 0.0],
     [0.0, 1.0, 0.0, 0.0],
     [0.0, 0.0, 1.0, 0.0],   # query carries s_i  (causally inert)
     [0.0, 0.0, 0.0, 1.0]],
    dtype=torch.float32, device=DEVICE,
)
_W_K_FULL = torch.tensor(
    [[2.0, 0.0, 0.0, 0.0],   # 2 x_j
     [0.0, 2.0, 0.0, 0.0],   # 2 y_j
     [0.0, 0.0, 0.0, -1.0],  # -1  (pairs with query const 1)
     [0.0, 0.0, -1.0, 0.0]], # -s_j  (the key-norm feature; NECESSARY)
    dtype=torch.float32, device=DEVICE,
)


def _phi(coords_t: torch.Tensor) -> torch.Tensor:
    """Fixed quadratic coordinate embedding: (N,2) -> (N,4) = [x, y, x^2+y^2, 1]."""
    x = coords_t[:, 0]
    y = coords_t[:, 1]
    s = x * x + y * y
    ones = torch.ones_like(x)
    return torch.stack([x, y, s, ones], dim=1)


def make_model_fn(mode: str = "full"):
    """Build a model_fn (NumPy in, NumPy out) computing one NN decode step on GPU.

    mode in {"full", "ablate_key_norm", "ablate_query_norm"}.
    """
    W_q = _W_Q_FULL.clone()
    W_k = _W_K_FULL.clone()
    if mode == "ablate_key_norm":
        W_k[3, :] = 0.0          # drop -s_j from the key
    elif mode == "ablate_query_norm":
        W_q[2, :] = 0.0          # drop s_i from the query (should be inert)
    elif mode != "full":
        raise ValueError(f"unknown mode {mode!r}")

    def model_fn(coords: np.ndarray, current_idx: int, visited: np.ndarray) -> np.ndarray:
        coords_t = torch.as_tensor(coords, dtype=torch.float32, device=DEVICE)  # (N,2)
        phi = _phi(coords_t)                       # (N,4)
        Q = phi @ W_q.T                            # (N,4)
        K = phi @ W_k.T                            # (N,4)
        q_i = Q[current_idx]                       # (4,)
        scores = K @ q_i                           # (N,) == Q_i . K_j  == -||c_i-c_j||^2 (full)
        return scores.detach().cpu().numpy()

    return model_fn


def _sweep_accs(task, mode: str):
    """Run task.evaluate with one variant and return its sweep + baseline."""
    payload = task.evaluate(make_model_fn(mode))
    return payload


def main():
    print("Running attention_tsp/pass_3 (QK = -squared-distance) on CUDA...")
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # --- Canonical payload: the FULL hand-set attention mechanism ------------
    payload = task.evaluate(make_model_fn("full"))
    payload["model_name"] = "qk_negdist_attention_handset"
    print("Full mechanism sweep:")
    for rec in payload["sweep"]:
        print(f"  n={rec['n']:>3}: nn_acc={rec['nn_accuracy']:.3f}  "
              f"tour_ratio={rec['tour_length_ratio']:.3f}")

    # --- Faithfulness/causal ablations (saved for the Demo tab) --------------
    variants = {}
    for mode in ("full", "ablate_key_norm", "ablate_query_norm"):
        p = task.evaluate(make_model_fn(mode))
        variants[mode] = [
            {"n": r["n"], "nn_accuracy": r["nn_accuracy"],
             "tour_length_ratio": r["tour_length_ratio"]}
            for r in p["sweep"]
        ]
        accs = ", ".join(f"n{r['n']}={r['nn_accuracy']:.2f}" for r in variants[mode])
        print(f"  [{mode:>18}] {accs}")

    random_baseline = [
        {"n": r["n"], "nn_accuracy": r["nn_accuracy"],
         "tour_length_ratio": r["tour_length_ratio"]}
        for r in payload["random_baseline"]
    ]

    ablation = {
        "n_cities_sweep": list(payload["n_cities_sweep"]),
        "variants": variants,
        "random_baseline": random_baseline,
        "weights": {
            "W_q_full": _W_Q_FULL.detach().cpu().numpy().tolist(),
            "W_k_full": _W_K_FULL.detach().cpu().numpy().tolist(),
        },
        "note": (
            "phi(x,y)=[x,y,x^2+y^2,1]; Q.K = -||c_i-c_j||^2. "
            "ablate_key_norm removes -s_j (NECESSARY -> collapses); "
            "ablate_query_norm removes s_i (constant in j -> INERT, accuracy "
            "unchanged)."
        ),
    }
    with open(run_dir / "ablation.json", "w") as f:
        json.dump(ablation, f, indent=2)
    print("Wrote ablation.json")

    record_benchmark(__file__, run_dir, payload)
    print("Results written to", run_dir)


if __name__ == "__main__":
    main()
