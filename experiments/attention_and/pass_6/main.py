"""
attention_and / pass_6 — "magnitude AND" head (hand-built, GPU).

Mechanism (no training, hand-set weights, all torch on cuda)
------------------------------------------------------------
A single attention head reads two QK probes off the residual stream:
    a = <r, q_A>,  b = <r, q_B>,  cos = <q_A, q_B>.
It forms a cosine-corrected magnitude  S = (a + b) / (2(1+cos)), an estimate of
*how many of the two features are present* (S in {0,1,2}), then thresholds it:
    logit = GAIN * sigmoid(STEEP * (S - 1.5)).
The gate fires only when S ~ 2 (both present) -> logical AND. Because the
*count* of features stays readable even when q_A and q_B merge (cos -> 1), the
boundary survives full superposition, where a per-feature product gate collapses
(its 2x2 Gram matrix is singular at cos=1).

Delta from base_model.py: the QKV projection reads q_A,q_B; the squared-ReLU MLP
is collapsed to one hand-set gating unit. Attention + one nonlinearity, no
training. main.py also runs two cheap ablations to show each piece matters.
"""
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"
GAIN, STEEP, THRESHOLD = 15.0, 6.0, 1.5


def _probe(q_A, q_B, residual):
    qA = torch.as_tensor(q_A, dtype=torch.float32, device=DEVICE)
    qB = torch.as_tensor(q_B, dtype=torch.float32, device=DEVICE)
    r = torch.as_tensor(residual, dtype=torch.float32, device=DEVICE)
    return r @ qA, r @ qB, torch.dot(qA, qB)


def and_head_fn(q_A, q_B, residual):
    a, b, cos = _probe(q_A, q_B, residual)
    S = (a + b) / (2.0 * (1.0 + cos))
    return (GAIN * torch.sigmoid(STEEP * (S - THRESHOLD))).detach().cpu().numpy()


def ablation_no_gate(q_A, q_B, residual):
    a, b, cos = _probe(q_A, q_B, residual)
    return ((a + b) / (2.0 * (1.0 + cos))).detach().cpu().numpy()


def ablation_product_gate(q_A, q_B, residual):
    a, b, cos = _probe(q_A, q_B, residual)
    denom = torch.clamp(1.0 - cos * cos, min=1e-6)
    alpha = (a - cos * b) / denom
    beta = (b - cos * a) / denom
    return ((alpha * beta) / 4.0).detach().cpu().numpy()


def _sharp(task, fn):
    return [rec["and_sharpness"] for rec in task.evaluate(fn)["sweep"]]


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    payload = task.evaluate(and_head_fn)
    record_benchmark(__file__, run_dir, payload)

    cos = payload["cos_AB_sweep"]
    ours = [r["and_sharpness"] for r in payload["sweep"]]
    base = [r["and_sharpness"] for r in payload["linear_baseline"]]

    data = {
        "cos_sweep": cos,
        "and_head": ours,
        "linear_baseline": base,
        "no_gate": _sharp(task, ablation_no_gate),
        "product_gate": _sharp(task, ablation_product_gate),
    }
    (run_dir / "ablations.json").write_text(json.dumps(data))

    batch = task.generate(seed=task.EVAL_SEED)
    snaps = {}
    for cp in (0.0, 1.0):
        i = next(j for j, c in enumerate(batch.cosines) if abs(c - cp) < 1e-9)
        a, b, c = _probe(batch.q_As[i], batch.q_Bs[i], batch.residuals[i])
        S = ((a + b) / (2.0 * (1.0 + c))).detach().cpu().numpy()
        snaps[f"cos_{cp:.1f}"] = {
            "S": S.tolist(),
            "label_and": batch.labels[i].astype(int).tolist(),
            "threshold": THRESHOLD,
        }
    (run_dir / "separation.json").write_text(json.dumps(snaps))

    print("ours:", [round(x, 3) for x in ours])
    print("base:", [round(x, 3) for x in base])


if __name__ == "__main__":
    main()
