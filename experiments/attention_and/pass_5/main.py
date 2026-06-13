"""
attention_and / pass_5 — "magnitude AND": a single thresholded attention head.

Hypothesis
----------
A clean logical AND over two query directions (q_A, q_B) in superposition is
implementable by ONE attention head provided the head applies a *threshold
nonlinearity* on top of its QK probe. The right quantity to threshold is the
TOTAL feature magnitude S = #features-present, NOT the per-feature estimates.

Why magnitude, not per-feature product?  As cos(q_A, q_B) -> 1 the two
directions merge and you can no longer tell A from B (the 2x2 Gram matrix is
singular). But you can still tell "both present" from "one present": each
present feature adds a fixed contribution, so the *combined* projection
distinguishes count 2 from count 1. Thresholding the magnitude therefore keeps
working at full superposition, where any per-feature product gate (pass_4)
collapses.

Mechanism (hand-set weights, no training)
-----------------------------------------
    a   = <residual, q_A>            # QK probe onto A
    b   = <residual, q_B>            # QK probe onto B
    cos = <q_A, q_B>
    S   = (a + b) / (2 * (1 + cos))  # cosine-corrected estimate of 1_A + 1_B
    logit = GAIN * sigmoid(STEEP * (S - 1.5))   # gate: fires only when S ~ 2

S in {0,1,2} for {neither, one, both}; the gate at 1.5 lights up only on AND.
The (1 + cos) correction keeps the threshold fixed at 1.5 across the whole
sweep, which is what buys superposition_robustness ~ 1.0.

Relation to base_model.py: this is the QKV projection (read q_A, q_B off the
residual) feeding a single gating nonlinearity — exactly the role the
squared-ReLU MLP plays in base_model.py, collapsed to one hand-set unit. No
extra layers; attention + one gate.

This file also evaluates four ablations and writes their sharpness sweeps as
artefacts so the Demo tab can show *why* each design choice matters.
"""
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback.

# Hand-set circuit constants.
GAIN = 15.0     # logit amplitude on the "both present" side of the gate
STEEP = 6.0     # gate sharpness (in units of S)
THRESHOLD = 1.5  # midpoint between "one feature" (S~1) and "both" (S~2)


def _probe(q_A, q_B, residual):
    """Move to GPU and compute the two QK projections + cosine. All torch/cuda."""
    qA = torch.as_tensor(q_A, dtype=torch.float32, device=DEVICE)
    qB = torch.as_tensor(q_B, dtype=torch.float32, device=DEVICE)
    r = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    cos = torch.dot(qA, qB)
    a = r @ qA
    b = r @ qB
    return a, b, cos


def and_head_fn(q_A: np.ndarray, q_B: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """THE mechanism: cosine-corrected magnitude gate. Runs on cuda."""
    a, b, cos = _probe(q_A, q_B, residual)
    S = (a + b) / (2.0 * (1.0 + cos))
    logits = GAIN * torch.sigmoid(STEEP * (S - THRESHOLD))
    return logits.detach().cpu().numpy()


# ---- Ablations (each isolates one design choice) ----------------------------

def ablation_no_threshold(q_A, q_B, residual):
    """Remove the gate: return the linear magnitude S itself (no nonlinearity).
    Should fall to the linear-baseline level — shows the gate is what sharpens."""
    a, b, cos = _probe(q_A, q_B, residual)
    S = (a + b) / (2.0 * (1.0 + cos))
    return S.detach().cpu().numpy()


def ablation_no_cosnorm(q_A, q_B, residual):
    """Keep the gate but drop the (1+cos) correction. Threshold drifts as the
    features overlap -> collapses at high cos. Shows the correction earns
    superposition robustness."""
    a, b, cos = _probe(q_A, q_B, residual)
    S = (a + b) / 2.0
    logits = GAIN * torch.sigmoid(STEEP * (S - THRESHOLD))
    return logits.detach().cpu().numpy()


def ablation_product_gate(q_A, q_B, residual):
    """The pass_4 idea: invert the Gram matrix, estimate alpha,beta separately,
    multiply. Sharp at low cos but the Gram matrix is singular at cos=1 ->
    robustness 0. Shown to contrast with the magnitude gate."""
    a, b, cos = _probe(q_A, q_B, residual)
    denom = torch.clamp(1.0 - cos * cos, min=1e-6)
    alpha = (a - cos * b) / denom
    beta = (b - cos * a) / denom
    logits = (alpha * beta) / 4.0
    return logits.detach().cpu().numpy()


def _sweep_sharpness(task, fn):
    """Return per-cosine and_sharpness list for a model_fn via the task evaluator."""
    payload = task.evaluate(fn)
    return [rec["and_sharpness"] for rec in payload["sweep"]]


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # --- headline: evaluate THE mechanism and record the benchmark payload ---
    payload = task.evaluate(and_head_fn)
    record_benchmark(__file__, run_dir, payload)

    cos_sweep = payload["cos_AB_sweep"]
    and_sharp = [rec["and_sharpness"] for rec in payload["sweep"]]
    base_sharp = [rec["and_sharpness"] for rec in payload["linear_baseline"]]

    # --- ablation sweeps (for the Demo viz) ---
    ablations = {
        "cos_sweep": cos_sweep,
        "and_head": and_sharp,
        "linear_baseline": base_sharp,
        "no_threshold": _sweep_sharpness(task, ablation_no_threshold),
        "no_cosnorm": _sweep_sharpness(task, ablation_no_cosnorm),
        "product_gate": _sweep_sharpness(task, ablation_product_gate),
    }
    (run_dir / "ablations.json").write_text(json.dumps(ablations, indent=2))

    # --- separation snapshots: gate-input S vs ground-truth AND at two cosines.
    # Saved so the demo can show the threshold cleanly splitting both/one/none.
    snaps = {}
    for cos_pick in (0.0, 1.0):
        batch = task.generate(seed=task.EVAL_SEED)
        # take the first seed-entry whose nominal cosine matches cos_pick
        idx = next(i for i, c in enumerate(batch.cosines) if abs(c - cos_pick) < 1e-9)
        qA, qB, res = batch.q_As[idx], batch.q_Bs[idx], batch.residuals[idx]
        a, b, cos = _probe(qA, qB, res)
        S = ((a + b) / (2.0 * (1.0 + cos))).detach().cpu().numpy()
        feat_A = res @ qA  # not labels; recover labels from batch
        label = batch.labels[idx].astype(int).tolist()
        snaps[f"cos_{cos_pick:.1f}"] = {
            "S": S.tolist(),
            "label_and": label,
            "threshold": THRESHOLD,
        }
    (run_dir / "separation.json").write_text(json.dumps(snaps, indent=2))

    metrics_summary = {
        "superposition_robustness": and_sharp[-1] / max(and_sharp[0], 1e-9),
        "and_sharpness_canonical": and_sharp[0],
        "lift_over_baseline": and_sharp[0] - base_sharp[0],
    }
    print("AND head sweep sharpness :", [round(x, 3) for x in and_sharp])
    print("linear baseline sharpness:", [round(x, 3) for x in base_sharp])
    print("no-cosnorm   sharpness   :", [round(x, 3) for x in ablations["no_cosnorm"]])
    print("product-gate sharpness   :", [round(x, 3) for x in ablations["product_gate"]])
    print("summary:", {k: round(v, 3) for k, v in metrics_summary.items()})
    print("artefacts ->", run_dir)


if __name__ == "__main__":
    main()
