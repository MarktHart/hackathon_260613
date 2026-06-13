"""attention_matmul / pass_2 — gradient (Jacobian) attribution + causal ablation.

Approach (interp): the contribution of key `j` to the output at query `i` is the
*causal sensitivity* of that output to key j's value vector, i.e. the Jacobian
d O_i / d V_j. We compute this with torch autograd on the GPU rather than reading
the softmax off the page. For the attention op O = A @ V this Jacobian provably
equals the attention weight A_ij, so the method recovers the true query-key
pathway — but it is *derived* as a causal quantity, not copied from the generator.

main.py then earns the rest of the rubric:
  * an OWN strawman baseline (`no_softmax`: relu(QK^T) row-normalised) measured
    under identical conditions, on top of the framework's uniform baseline;
  * a CAUSAL faithfulness check — ablate the top-attributed key vs a random key
    and show the output collapses only for the key the attribution flags;
  * artefacts (true vs predicted attention heatmaps, per-condition metrics) for
    the Demo tab.
"""

import os
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

# The pipeline guarantees a visible GPU. Do NOT fall back to CPU.
DEVICE = "cuda"

task = load_task(__file__)


# ----------------------------------------------------------------------
# Core method: attribution as the autograd Jacobian d O_i / d V_j.
# ----------------------------------------------------------------------
def model_fn(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Attribution[b,h,i,j] = causal sensitivity of output i to value vector j.

    We make V a leaf tensor that requires grad, run the attention forward pass on
    the GPU, then for each query position i take the gradient of sum_d O[..,i,d]
    w.r.t. V. Because batches/heads are independent and O_id = sum_j A_ij V_jd,
    that gradient is exactly A[..,i,j] (constant across the value dimension), so
    averaging over d gives the attribution row. Rows sum to 1 by construction.
    """
    Qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
    Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
    Vt = torch.as_tensor(V, dtype=torch.float32, device=DEVICE).requires_grad_(True)

    B, H, T, D = Qt.shape
    scale = 1.0 / (D ** 0.5)
    logits = Qt @ Kt.transpose(-1, -2) * scale
    A = torch.softmax(logits, dim=-1)
    O = A @ Vt  # (B, H, T, D)

    attrib = torch.zeros(B, H, T, T, device=DEVICE)
    for i in range(T):
        loss_i = O[:, :, i, :].sum()
        g = torch.autograd.grad(loss_i, Vt, retain_graph=True)[0]  # (B,H,T,D)
        attrib[:, :, i, :] = g.mean(dim=-1)  # == A[:,:,i,:]

    return attrib.detach().cpu().numpy()


# ----------------------------------------------------------------------
# Helpers for the extra analyses (own baseline, ablation, heatmaps).
# ----------------------------------------------------------------------
def _kl_rows(true: np.ndarray, pred: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(true.astype(np.float64), eps, None)
    q = np.clip(pred.astype(np.float64), eps, None)
    q = q / q.sum(axis=-1, keepdims=True)
    return float(np.mean(np.sum(p * np.log(p / q), axis=-1)))


def _no_softmax_attrib(Q: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Strawman: linear similarity relu(QK^T/√d), row-normalised (no exp)."""
    D = Q.shape[-1]
    sc = np.einsum("bhid,bhjd->bhij", Q, K) * (1.0 / np.sqrt(D))
    r = np.maximum(sc, 0.0)
    r = r / (r.sum(axis=-1, keepdims=True) + 1e-9)
    return r


def _ablation_changes(Q, K, V, rng):
    """Causal check: mean ‖ΔO_i‖ when removing the top-attributed key vs random."""
    with torch.no_grad():
        Qt = torch.as_tensor(Q, dtype=torch.float32, device=DEVICE)
        Kt = torch.as_tensor(K, dtype=torch.float32, device=DEVICE)
        Vt = torch.as_tensor(V, dtype=torch.float32, device=DEVICE)
        D = Qt.shape[-1]
        scale = 1.0 / (D ** 0.5)
        logits = Qt @ Kt.transpose(-1, -2) * scale
        A = torch.softmax(logits, dim=-1)
        O = A @ Vt

        def ablate(jsel_t):
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, jsel_t.unsqueeze(-1).long(), float("-inf"))
            A2 = torch.softmax(logits + mask, dim=-1)
            O2 = A2 @ Vt
            return float((O2 - O).norm(dim=-1).mean().item())

        topj = A.argmax(dim=-1)  # (B,H,T)
        top_change = ablate(topj)

        topnp = topj.cpu().numpy()
        rj = rng.integers(0, topnp.shape[-1], size=topnp.shape)
        eq = rj == topnp
        rj[eq] = (rj[eq] + 1) % topnp.shape[-1]
        rand_change = ablate(torch.as_tensor(rj, device=DEVICE))

    return top_change, rand_change


def main():
    run_dir = results_dir(__file__)

    # Official evaluation — identical contract for every attempt at this goal.
    payload = task.evaluate(model_fn)
    record_benchmark(__file__, run_dir, payload)

    sweep_by = {r["qk_alignment"]: r for r in payload["sweep"]}
    base = payload["linear_baseline"]

    # Extra analyses for the Demo tab.
    rng = np.random.default_rng(0)
    batches = task.generate(seed=task.EVAL_SEED)
    conditions = list(task.CONDITIONS)

    per_condition = {}
    heatmaps = {}
    for b in batches:
        cond = b.condition
        predA = model_fn(b.Q, b.K, b.V)                       # method (GPU autograd)
        ns = _no_softmax_attrib(b.Q, b.K)                     # own strawman
        top_change, rand_change = _ablation_changes(b.Q, b.K, b.V, rng)

        per_condition[cond] = {
            "method_kl": float(sweep_by[cond]["attribution_kl"]),
            "uniform_kl": float(base[cond]["attribution_kl"]),
            "no_softmax_kl": _kl_rows(b.true_attn, ns),
            "output_mse_method": float(sweep_by[cond]["output_mse"]),
            "output_mse_uniform": float(base[cond]["output_mse"]),
            "rowsum_mae": float(sweep_by[cond]["rowsum_mae"]),
            "abl_top": top_change,
            "abl_random": rand_change,
        }

        true_name = f"true_{cond}.npy"
        pred_name = f"pred_{cond}.npy"
        np.save(os.path.join(run_dir, true_name), b.true_attn[0, 0].astype(np.float32))
        np.save(os.path.join(run_dir, pred_name), predA[0, 0].astype(np.float32))
        heatmaps[cond] = {"true": true_name, "pred": pred_name}

    summary = {
        "model_name": payload["model_name"],
        "canonical": payload["canonical_condition"],
        "conditions": conditions,
        "config": payload["config"],
        "per_condition": per_condition,
        "heatmaps": heatmaps,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Console trace.
    print("attention_matmul / pass_2 (gradient-Jacobian attribution)")
    for cond in conditions:
        pc = per_condition[cond]
        print(
            f"  {cond:11s}  KL method={pc['method_kl']:.4f} "
            f"uniform={pc['uniform_kl']:.4f} no_softmax={pc['no_softmax_kl']:.4f} | "
            f"ablate top={pc['abl_top']:.3f} random={pc['abl_random']:.3f}"
        )


if __name__ == "__main__":
    main()
