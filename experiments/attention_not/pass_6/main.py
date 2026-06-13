"""Hand-built inhibitory NOT head via dual-basis read-out of superposed features.

Attempt type: hand_built (no learning, no trained model — pure circuit design).

The goal asks whether a single attention head can compute `A AND NOT B`
*while the attend-direction `e_A` and suppress-direction `e_B` are in
superposition* (cos(theta) > 0).  The previous attempt (pass_5) sidestepped
this by reading the oracle labels `feat_A/feat_B` and forcing per-sequence keys
with a pseudo-inverse — it never used `e_A/e_B`, so its robustness was an
artefact of being invariant to cos(theta).

This attempt engages the superposition axis head-on:

1.  WRITE the features into the residual stream *in superposition*: the query
    position's residual is
        x_q = feat_A * e_A + feat_B * e_B + p_bias
    where `e_A, e_B` are the task's unit directions at cosine theta and
    `p_bias` is a fixed direction orthogonal to both (a constant-bias channel).
    Features enter ONLY through this superposed write — never as oracle keys.

2.  READ them back with a single real attention head built from the task's own
    random `W_Q, W_K`.  The query→A logit is `x_q^T (W_Q W_K^T) x_A`, where the
    A-token embedding `x_A` is the ONE hand-set vector of the circuit.  We solve
    for `x_A` so that, *through the real QK matrix*,
        e_A^T (W_Q W_K^T) x_A = +ALPHA      (attend when A present)
        e_B^T (W_Q W_K^T) x_A = -BETA       (suppress when B present)
        p^T   (W_Q W_K^T) x_A = +DELTA      (negative resting bias)
    so the logit is exactly  ALPHA*feat_A - BETA*feat_B + DELTA — a clean NOT.

    Solving these three dot-product constraints *is* the dual (reciprocal) basis
    of {e_A, e_B}: it implicitly inverts the Gram matrix [[1,cos],[cos,1]] to
    cancel the cross-talk that superposition injects.  The "correction" is not
    an add-on — it is exactly what solving the QK circuit for a clean NOT forces.

3.  COST OF SUPERPOSITION.  As cos(theta) -> 1 the two constraint rows become
    parallel, the Gram inverse blows up, and the min-norm `||x_A||` required to
    keep the NOT sharp grows.  We record this norm so the visualisation can show
    *why* superposition is hard, not just that we survive it.

4.  ABLATION / STRAWMAN.  A "naive" head reads with the RAW directions
    (r = ALPHA*e_A - BETA*e_B + DELTA*p) instead of the dual basis — i.e. the
    dual-basis correction is knocked out.  Its logit becomes
        (ALPHA - BETA*cos)*feat_A + (ALPHA*cos - BETA)*feat_B + DELTA,
    so the attend term collapses as cos grows: at cos=0 it equals the corrected
    head, by cos=0.8 the query barely attends to A at all.  This is the causal
    knockout showing the dual basis is load-bearing.

All compute is torch on CUDA.  The submitted payload is the *corrected* head;
the naive sweep and the norm curve are saved alongside for the Demo tab.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"

# Logit budget for the hand-set NOT.  BETA > ALPHA => B fully overrides A.
ALPHA = 12.0   # attend strength when feat_A present
BETA = 18.0    # suppress strength when feat_B present
DELTA = -4.0   # negative resting bias -> low false-attend when A absent

QUERY_POS = 2
A_POS = 0


# --------------------------------------------------------------------------- #
#  Circuit construction (all on GPU)                                          #
# --------------------------------------------------------------------------- #
def _geometry(batch):
    """Move the task geometry to the GPU and build the orthogonal bias channel."""
    e_A = torch.as_tensor(np.asarray(batch.e_A), dtype=torch.float32, device=DEVICE)
    e_B = torch.as_tensor(np.asarray(batch.e_B), dtype=torch.float32, device=DEVICE)
    W_Q = torch.as_tensor(np.asarray(batch.W_Q), dtype=torch.float32, device=DEVICE)
    W_K = torch.as_tensor(np.asarray(batch.W_K), dtype=torch.float32, device=DEVICE)
    feat_A = torch.as_tensor(np.asarray(batch.feat_A), dtype=torch.float32, device=DEVICE)
    feat_B = torch.as_tensor(np.asarray(batch.feat_B), dtype=torch.float32, device=DEVICE)
    d_model = W_Q.shape[0]

    # Deterministic bias direction, then project OUT of span{e_A, e_B} so the
    # resting bias does not contaminate the feature read-outs.
    idx = torch.arange(d_model, device=DEVICE, dtype=torch.float32)
    p = torch.cos(0.7 * idx + 0.3)
    E = torch.stack([e_A, e_B])                       # (2, d)
    G = E @ E.T                                        # (2, 2) Gram
    coef = torch.linalg.solve(G, E @ p)                # (2,)
    p = p - coef @ E                                   # remove span{e_A,e_B}
    p = p / p.norm()
    return e_A, e_B, W_Q, W_K, feat_A, feat_B, p, d_model


def _solve_xA(e_A, e_B, p, W_Q, W_K):
    """Hand-set A-token embedding that yields logit = A*ALPHA - B*BETA + DELTA
    through the real QK matrix M = W_Q W_K^T.  This is the dual basis."""
    M = W_Q @ W_K.T                                    # (d, d)
    # Constraint rows live in residual space: (M^T e_*)^T x_A = target.
    P = torch.stack([M.T @ e_A, M.T @ e_B, M.T @ p])   # (3, d)
    b = torch.tensor([ALPHA, -BETA, DELTA], dtype=torch.float32, device=DEVICE)
    # Min-norm solution of the under-determined system P x = b.
    x_A = P.T @ torch.linalg.solve(P @ P.T, b)         # (d,)
    return x_A, M


def _query_residual(feat_A, feat_B, e_A, e_B, p):
    """Superposed write: x_q = A*e_A + B*e_B + p_bias  (n, d)."""
    return feat_A[:, None] * e_A[None, :] + feat_B[:, None] * e_B[None, :] + p[None, :]


def corrected_model_fn(batch) -> dict:
    """Real attention head: dual-basis read-out of superposed features."""
    e_A, e_B, W_Q, W_K, feat_A, feat_B, p, d = _geometry(batch)
    x_A, _ = _solve_xA(e_A, e_B, p, W_Q, W_K)
    n = feat_A.shape[0]
    seq_len = 4

    x_q = _query_residual(feat_A, feat_B, e_A, e_B, p)        # (n, d)
    Xq = torch.zeros(n, seq_len, d, device=DEVICE)
    Xk = torch.zeros(n, seq_len, d, device=DEVICE)
    Xq[:, QUERY_POS, :] = x_q
    Xk[:, A_POS, :] = x_A[None, :]                            # only the A-token carries a key

    Q = Xq @ W_Q                                             # (n, 4, d_head)
    K = Xk @ W_K
    logits = Q @ K.transpose(1, 2)                          # (n, 4, 4); query row = [logit_A, 0, 0, 0]
    attn = torch.softmax(logits, dim=-1)
    return {"attn_weights": attn.detach().cpu().numpy().astype(np.float64)}


def naive_model_fn(batch) -> dict:
    """ABLATION: read with raw e_A/e_B directions (dual-basis correction removed).
    Logit = x_q . (ALPHA e_A - BETA e_B + DELTA p); cross-talk corrupts it as cos grows."""
    e_A, e_B, W_Q, W_K, feat_A, feat_B, p, d = _geometry(batch)
    n = feat_A.shape[0]
    seq_len = 4

    r = ALPHA * e_A - BETA * e_B + DELTA * p                 # raw-direction read-out
    x_q = _query_residual(feat_A, feat_B, e_A, e_B, p)       # (n, d)
    logit_A = x_q @ r                                        # (n,)
    logits = torch.zeros(n, seq_len, seq_len, device=DEVICE)
    logits[:, QUERY_POS, A_POS] = logit_A
    attn = torch.softmax(logits, dim=-1)
    return {"attn_weights": attn.detach().cpu().numpy().astype(np.float64)}


# --------------------------------------------------------------------------- #
#  Viz helpers                                                                #
# --------------------------------------------------------------------------- #
def _norm_cost(task, cos_list, seed=1234):
    """Min-norm ||x_A|| required to keep the NOT sharp, per cos slice.

    Held at a FIXED seed so only cos(theta) varies (same W_Q/W_K and e_A across
    slices) — this isolates the superposition geometry from the per-slice
    randomness the evaluator uses, giving a clean cost-of-separation curve."""
    norms = []
    for cos in cos_list:
        batch = task.generate(seed=seed, cos_theta=cos)
        e_A, e_B, W_Q, W_K, _, _, p, _ = _geometry(batch)
        x_A, _ = _solve_xA(e_A, e_B, p, W_Q, W_K)
        norms.append(float(x_A.norm().item()))
    return norms


def _combo_rows(attn, feat_A, feat_B):
    out = {}
    for a, b, name in [(1, 0, "A1B0"), (1, 1, "A1B1"), (0, 0, "A0B0"), (0, 1, "A0B1")]:
        idx = np.where((feat_A == a) & (feat_B == b))[0]
        out[name] = attn[idx[0], QUERY_POS, :].tolist() if len(idx) else [0.0, 0.0, 0.0, 0.0]
    return out


def main() -> None:
    task = load_task(__file__)

    # --- submitted payload: the corrected (dual-basis) head ---
    payload = task.evaluate(corrected_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)

    # --- ablation sweep for the Demo tab ---
    naive_payload = task.evaluate(naive_model_fn)

    cos_list = [r["cos"] for r in payload["sweep"]]
    norms = _norm_cost(task, cos_list)

    # --- attention examples at the easy and hard ends of the sweep ---
    examples = {}
    for cos in (cos_list[0], cos_list[-1]):
        i = cos_list.index(cos)
        batch = task.generate(seed=1234 + i, cos_theta=cos)
        fa = np.asarray(batch.feat_A)
        fb = np.asarray(batch.feat_B)
        examples[f"{cos:.1f}"] = {
            "corrected": _combo_rows(corrected_model_fn(batch)["attn_weights"], fa, fb),
            "naive": _combo_rows(naive_model_fn(batch)["attn_weights"], fa, fb),
        }

    viz = {
        "cos": cos_list,
        "corrected": {
            "not_sharpness": [r["not_sharpness"] for r in payload["sweep"]],
            "suppression_gap": [r["suppression_gap"] for r in payload["sweep"]],
            "attend_specificity": [r["attend_specificity"] for r in payload["sweep"]],
        },
        "naive": {
            "not_sharpness": [r["not_sharpness"] for r in naive_payload["sweep"]],
            "suppression_gap": [r["suppression_gap"] for r in naive_payload["sweep"]],
            "attend_specificity": [r["attend_specificity"] for r in naive_payload["sweep"]],
        },
        "baseline": {
            "not_sharpness": [r["not_sharpness"] for r in payload["baseline"]],
        },
        "xA_norm": norms,
        "examples": examples,
        "constants": {"ALPHA": ALPHA, "BETA": BETA, "DELTA": DELTA},
    }
    with (run_dir / "viz_data.json").open("w") as f:
        json.dump(viz, f, indent=2)

    # --- console report ---
    print("=== corrected (dual-basis) head — submitted payload ===")
    for rec in payload["sweep"]:
        print(f"  cos={rec['cos']:.1f}: sharp={rec['not_sharpness']:.3f} "
              f"gap={rec['suppression_gap']:.3f} spec={rec['attend_specificity']:.3f}")
    print("=== naive (ablated dual-basis) head ===")
    for rec, c in zip(naive_payload["sweep"], cos_list):
        print(f"  cos={c:.1f}: sharp={rec['not_sharpness']:.3f} gap={rec['suppression_gap']:.3f}")
    print("=== ||x_A|| norm cost vs cos ===")
    print("  " + ", ".join(f"{c:.1f}:{nrm:.1f}" for c, nrm in zip(cos_list, norms)))
    print(f"saved viz_data.json -> {run_dir}")


if __name__ == "__main__":
    main()
