"""attention_xor / pass_5 — hand-built single-head attention + ReLU-MLP XOR circuit.

This is a *hand_built* attempt (no training). It is the smallest delta from
``base_model.py``: a single attention head feeding a two-unit ReLU MLP read-out,
with every weight set by hand. All compute runs on CUDA.

Mechanism (why XOR is non-linear and how attention captures it)
---------------------------------------------------------------
The four input cells (A,B) -> XOR are  (0,0)->0  (0,1)->1  (1,0)->1  (1,1)->0.
No linear probe over the one-hot bits separates them (XOR is the canonical
not-linearly-separable function). The circuit splits the problem in two:

  1. ATTENTION pools the two bits into their SUM.  The CLS token's query points
     at the "A-type" and "B-type" key channels; A_tok and B_tok both score
     high, CLS/SEP score zero, so softmax puts ~0.5 weight on each of A_tok and
     B_tok. The pooled value channel is therefore  0.5*(A+B), and a x2 read-out
     gives  s = A + B  in {0, 1, 2}.

  2. A 2-unit ReLU MLP turns the sum into a BUMP that fires only at s == 1:
         logit = 0.5 - relu(s - 1) - relu(1 - s)
         s=0 -> 0.5 - 0 - 1 = -0.5   (XOR 0, pred 0)  OK
         s=1 -> 0.5 - 0 - 0 = +0.5   (XOR 1, pred 1)  OK
         s=2 -> 0.5 - 1 - 0 = -0.5   (XOR 0, pred 0)  OK
     The two ReLUs are the non-linearity the linear floor cannot express:
     XOR = 1 iff A+B is *exactly* 1, a band the linear probe must give up on.

Faithfulness / ablation (built in)
----------------------------------
``ablation_fn`` keeps the *same* attention pooling but drops one ReLU unit:
``logit = 0.5 - relu(s - 1)``. That degrades the bump into the monotone
threshold NAND ( predict 1 for s in {0,1} ), which is linearly separable and
therefore cannot beat the best-linear-probe floor. Knocking out a single hidden
unit collapses the mechanism back onto the baseline — direct evidence the XOR
behaviour lives in the ReLU MLP, not in the attention pooling alone.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

# ---- Vocabulary (see goal README): 0=CLS 1=A0 2=A1 3=B0 4=B1 5=SEP ----
VOCAB = 6
D = 16

# Residual-stream channel assignments (hand-picked, orthogonal).
BIT_DIM = 0     # carries the feature bit (A or B): A1/B1 -> 1, A0/B0 -> 0
CLS_DIM = 1     # marks the CLS token so it can emit the pooling query
ATYPE_DIM = 2   # "I am an A token" key channel
BTYPE_DIM = 3   # "I am a B token" key channel

ATTN_SCALE = 30.0  # query magnitude -> softmax puts ~0.5 on each of A_tok,B_tok


def build_weights():
    """Return hand-set (E, Wq, Wk, Wv) as CUDA tensors."""
    E = torch.zeros((VOCAB, D), device=DEVICE)
    E[0, CLS_DIM] = 1.0                 # CLS marker
    E[1, ATYPE_DIM] = 1.0               # A0: bit 0, type A
    E[2, BIT_DIM] = 1.0                 # A1: bit 1, type A
    E[2, ATYPE_DIM] = 1.0
    E[3, BTYPE_DIM] = 1.0               # B0: bit 0, type B
    E[4, BIT_DIM] = 1.0                 # B1: bit 1, type B
    E[4, BTYPE_DIM] = 1.0
    # SEP (id 5): all zeros.

    # Query: CLS marker -> strong attention onto both type channels.
    Wq = torch.zeros((D, D), device=DEVICE)
    Wq[ATYPE_DIM, CLS_DIM] = ATTN_SCALE
    Wq[BTYPE_DIM, CLS_DIM] = ATTN_SCALE

    # Key: pass the type channels through unchanged.
    Wk = torch.zeros((D, D), device=DEVICE)
    Wk[ATYPE_DIM, ATYPE_DIM] = 1.0
    Wk[BTYPE_DIM, BTYPE_DIM] = 1.0

    # Value: pass the bit channel through unchanged.
    Wv = torch.zeros((D, D), device=DEVICE)
    Wv[BIT_DIM, BIT_DIM] = 1.0

    return E, Wq, Wk, Wv


def _forward(tokens):
    """Run the single attention head. Returns (out, attn) CUDA tensors.

    out:  (N, 4, D) per-position residual after attention.
    attn: (N, 4, 4) softmax attention weights.
    """
    E, Wq, Wk, Wv = build_weights()
    idx = torch.as_tensor(np.asarray(tokens), dtype=torch.long, device=DEVICE)
    if idx.ndim == 1:
        idx = idx[None, :]
    emb = E[idx]                                   # (N, 4, D)
    q = emb @ Wq.t()
    k = emb @ Wk.t()
    v = emb @ Wv.t()
    scores = q @ k.transpose(-1, -2)               # (N, 4, 4)
    attn = F.softmax(scores, dim=-1)
    out = attn @ v                                 # (N, 4, D)
    return out, attn


def _logits_from_out(out, relus: int = 2):
    """Read s = A+B off the CLS value channel, then apply the ReLU bump."""
    cls_bit = out[:, 0, BIT_DIM]                   # ~ 0.5 * (A + B)
    s = 2.0 * cls_bit                              # ~ A + B in {0,1,2}
    if relus == 2:
        logit = 0.5 - F.relu(s - 1.0) - F.relu(1.0 - s)
    else:  # ablation: drop one hidden unit -> monotone NAND, linearly separable
        logit = 0.5 - F.relu(s - 1.0)
    return s, logit


def model_fn(tokens: np.ndarray) -> np.ndarray:
    """Full circuit: attention pooling + 2-unit ReLU MLP. Predicts XOR=1 iff logit>0."""
    out, _ = _forward(tokens)
    _, logit = _logits_from_out(out, relus=2)
    return logit.detach().cpu().numpy().astype(np.float64)


def ablation_fn(tokens: np.ndarray) -> np.ndarray:
    """Ablated circuit: same attention, one ReLU removed -> collapses to baseline."""
    out, _ = _forward(tokens)
    _, logit = _logits_from_out(out, relus=1)
    return logit.detach().cpu().numpy().astype(np.float64)


def explain(a: int, b: int) -> dict:
    """Run one (A,B) example and expose the internals for the demo view."""
    tokens = np.array([[0, a + 1, b + 3, 5]], dtype=np.int64)
    out, attn = _forward(tokens)
    s, logit = _logits_from_out(out, relus=2)
    _, abl = _logits_from_out(out, relus=1)
    w = attn[0, 0].detach().cpu().numpy()          # CLS attention over 4 positions
    return {
        "A": int(a),
        "B": int(b),
        "xor": int(a ^ b),
        "s": float(s[0].item()),
        "attn_cls": [float(x) for x in w.tolist()],
        "logit": float(logit[0].item()),
        "pred": int(logit[0].item() > 0),
        "ablation_logit": float(abl[0].item()),
        "ablation_pred": int(abl[0].item() > 0),
    }


def truth_table_records() -> list[dict]:
    """All four input cells with internals, for the demo table / bar plot."""
    recs = []
    for a in (0, 1):
        for b in (0, 1):
            e = explain(a, b)
            recs.append(
                {
                    "cell": f"A={a},B={b}",
                    "A": a,
                    "B": b,
                    "s (=A+B)": round(e["s"], 3),
                    "attn_A": round(e["attn_cls"][1], 3),
                    "attn_B": round(e["attn_cls"][2], 3),
                    "xor": e["xor"],
                    "logit": round(e["logit"], 3),
                    "pred": e["pred"],
                    "ablation_logit": round(e["ablation_logit"], 3),
                    "ablation_pred": e["ablation_pred"],
                }
            )
    return recs


def main():
    task = load_task(__file__)
    payload = task.evaluate(model_fn)
    run_dir = results_dir(__file__)

    # Diagnostics for the demo / faithfulness story.
    abl_payload = task.evaluate(ablation_fn)
    diag = {
        "truth_table": truth_table_records(),
        "model_sweep": [
            {"p": r["p"], "accuracy": r["accuracy"], "baseline": r["baseline_accuracy"]}
            for r in payload["sweep"]
        ],
        "ablation_sweep": [
            {"p": r["p"], "accuracy": r["accuracy"], "baseline": r["baseline_accuracy"]}
            for r in abl_payload["sweep"]
        ],
    }
    with open(os.path.join(run_dir, "diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    # Console summary.
    can = next(r for r in payload["sweep"] if abs(r["p"] - 0.5) < 1e-9)
    print(
        f"[pass_5] canonical p=0.5  acc={can['accuracy']:.3f}  "
        f"base={can['baseline_accuracy']:.3f}"
    )
    print(f"[pass_5] worst-slice acc = {min(r['accuracy'] for r in payload['sweep']):.3f}")


if __name__ == "__main__":
    main()
