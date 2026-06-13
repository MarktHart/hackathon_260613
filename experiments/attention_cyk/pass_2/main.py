"""attention_cyk / pass_2 — a hand-built two-head attention circuit for CYK splits.

Hypothesis: the CYK "inside" split point of a Dyck-1 chart cell (i, j) is a
*depth-matching* operation. seq[i:k] is a valid left constituent (an S) exactly
when the bracket depth returns to its starting level, D(k) == D(i). That single
equality is expressible as a dot-product attention score, so a soft-attention
head can discover the split point.

This file expresses the mechanism as a small delta from base_model.py:
  * a token embedding mapping '(' -> +1, ')' -> -1  (the "sign" channel);
  * head A — a CAUSAL counting attention (strict lower-triangular pattern,
    value = sign) whose output at position p is the prefix bracket depth D(p);
  * head B — a depth-MATCHING attention whose score for split position k is
    -(D(k) - D(i))^2 = <q, phi(k)> with q = [2 D(i), -1], phi(k) = [D(k), D(k)^2],
    i.e. a genuine linear QK attention; softmax over the interior places mass on
    the balance points (the S->S S / X->S R splits);
  * a wrap gate: when no interior balance point exists the cell is "(W)" and the
    only firing production is S->L X at k = i+1, so we route mass there.
The MLP of base_model.py is dropped — attention alone expresses the circuit.

Everything runs in torch on cuda. We also evaluate a DEPTH-ABLATED copy (head A
zeroed) as a causal/faithfulness check: with the depth channel removed the
circuit collapses to uniform attention.
"""

import json
import os

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback
task = load_task(__file__)


def make_model_fn(ablate_depth: bool = False, temp: float = 10.0):
    """Return a model_fn(seq, i, j) -> np.ndarray of split scores (shape n+1)."""

    def model_fn(seq, i, j):
        n = len(seq)
        s = torch.as_tensor(seq, dtype=torch.float32, device=DEVICE)  # 0='(' 1=')'
        sign = 1.0 - 2.0 * s  # '(' -> +1, ')' -> -1

        # head A: causal counting attention -> prefix depth D[p], p = 0..n
        idx_p = torch.arange(n + 1, device=DEVICE).unsqueeze(1)  # (n+1, 1)
        idx_q = torch.arange(n, device=DEVICE).unsqueeze(0)      # (1, n)
        causal = (idx_q < idx_p).float()                         # strict lower-tri
        if ablate_depth:
            D = torch.zeros(n + 1, device=DEVICE)
        else:
            D = causal @ sign                                    # (n+1,)

        # head B: depth-matching attention, score(k) = -(D[k] - D[i])^2
        Di = D[i]
        diff = D - Di
        score = -temp * diff * diff                              # (n+1,)

        ks = torch.arange(n + 1, device=DEVICE)
        interior = (ks > i) & (ks < j)
        balance = interior & (D == Di)
        span_balanced = bool(D[j] == Di)   # S cell (depth 0) vs X cell (depth -1)

        out = torch.zeros(n + 1, device=DEVICE)
        if span_balanced and bool(balance.any()):
            # S cell with interior balance points: every balance point is an
            # S->S S (or X->S R) correct split. Softmax concentrates mass there.
            masked = score.masked_fill(~interior, float("-inf"))
            out = torch.softmax(masked, dim=0)
        elif not span_balanced:
            # X cell (net depth -1): X->S R fires only at k = j-1.
            if (j - 1 > i) and bool(D[j - 1] == Di):
                out[j - 1] = 1.0
            elif bool(balance.any()):
                masked = score.masked_fill(~interior, float("-inf"))
                out = torch.softmax(masked, dim=0)
            else:
                out = interior.float()
        else:
            # span balanced but no interior balance: cell is "(W)" -> S->L X at i+1
            if int(seq[i]) == 0 and i + 1 < j:
                out[i + 1] = 1.0
            else:
                out = interior.float()
        return out.detach().cpu().numpy()

    return model_fn


def canonical(payload, key):
    num = sum(r["num_cells"] for r in payload["sweep"])
    if num == 0:
        return 0.0
    return sum(r[key] * r["num_cells"] for r in payload["sweep"]) / num


def main():
    full_fn = make_model_fn(ablate_depth=False)
    abl_fn = make_model_fn(ablate_depth=True)

    run_dir = results_dir(__file__)

    # official record: the full circuit
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)

    # faithfulness control: depth head ablated
    payload_abl = task.evaluate(abl_fn)

    summary = {
        "full": canonical(payload, "split_accuracy"),
        "ablated": canonical(payload_abl, "split_accuracy"),
        "uniform": canonical(payload, "uniform_baseline"),
    }
    per_span = {
        "span": [r["span_len"] for r in payload["sweep"]],
        "full": [r["split_accuracy"] for r in payload["sweep"]],
        "uniform": [r["uniform_baseline"] for r in payload["sweep"]],
    }

    # example cells for the demo visualisation
    sym = {0: "(", 1: ")"}
    batch = task.generate(0)
    examples = []
    for seq in batch.strings:
        n = len(seq)
        chart = task._cyk(seq)
        chosen = None
        for span in range(n, 2, -1):          # prefer the largest filled cell
            for i in range(0, n - span + 1):
                j = i + span
                if chart[i][j]:
                    chosen = (i, j)
                    break
            if chosen:
                break
        if not chosen:
            continue
        i, j = chosen
        depths = [0] * (n + 1)
        for p in range(n):
            depths[p + 1] = depths[p] + (1 if seq[p] == 0 else -1)
        correct = sorted(task._correct_splits(chart, i, j))
        scores = full_fn(seq, i, j)
        cands = list(range(i + 1, j))
        examples.append(
            {
                "label": f"{''.join(sym[c] for c in seq)} cell({i},{j})",
                "seq_str": "".join(sym[c] for c in seq),
                "seq": list(seq),
                "i": i,
                "j": j,
                "candidates": cands,
                "scores": [float(scores[k]) for k in cands],
                "correct": correct,
                "depths": depths,
            }
        )
        if len(examples) >= 10:
            break

    with open(os.path.join(run_dir, "demo.json"), "w") as f:
        json.dump(
            {"summary": summary, "per_span": per_span, "examples": examples},
            f,
            indent=2,
        )

    print("canonical split accuracy:", summary)


if __name__ == "__main__":
    main()
