"""attention_cyk / pass_3 — ONE pure-attention head for the CYK split point.

pass_2 reached ~1.0 but the jury flagged *hand-coded Python routing* (an
``if span_balanced / elif X-cell / else wrap`` dispatch) as diluting the
"pure attention" claim. pass_3 removes ALL of that: a single attention head
with a genuine linear QK score solves every cell type with no data-dependent
branching whatsoever.

The mechanism
-------------
A causal counting head (strict lower-triangular attention over the sign
embedding ``( -> +1``, ``) -> -1``) yields the prefix bracket depth ``D(p)``.
The split head then scores candidate split ``k`` for query cell ``(i, j)`` as

    score(k) = -T * (D(k) - D(i))**2  +  beta * (D(i) + 0.5 - D(k)) * k

which is exactly a bilinear attention score <q(i), phi(k)> with
    phi(k) = [ D(k), D(k)**2, k, k*D(k) ]
    q(i)   = [ 2T*D(i) + beta*(D(i)+0.5),  -T,  ... ,  -beta ]
(the D(i)**2 piece is an additive constant). One head, no gate, no routing.

Why this single score is correct for EVERY filled (S or X) query cell:
  * Interior depth of a filled cell never dips below D(i), so only depth
    levels D(i), D(i)+1, ... appear inside the span.
  * The quadratic makes EVERY point at level D(i) outrank EVERY point at a
    higher level (the linear term is >=0 at level D(i) and <=0 above it, and
    the quadratic adds T on top) -- so the two regimes never interfere, for
    ANY beta.
  * Among level-D(i) points the bias is +beta*0.5*k -> the LATEST one wins.
      - X cells: the only correct split is k=j-1, always the latest D(i) point.
      - S cells with balance points: every level-D(i) point is a correct S->S S
        split, so the latest is correct.
  * If there is NO level-D(i) interior point the cell is a single wrap "(S)";
    the minimum level is D(i)+1 and the bias there is -beta*0.5*k -> the
    EARLIEST one wins = k=i+1, which is exactly the S->L X split.

Everything runs in torch on cuda. We also evaluate two controls:
  * ``no_position`` (beta=0)  -> pure depth-matching; spreads mass over all
    balance points, so X cells and multi-balance cells degrade.
  * ``depth_ablated`` (D:=0)  -> score is position-only; loses S cells.
Both knock out a named ingredient and watch accuracy fall -- faithfulness.
"""

import json
import os

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback
task = load_task(__file__)

T_QUAD = 50.0   # depth-matching sharpness
BETA = 10.0     # position tie-break strength (any beta is level-safe)


def make_model_fn(ablate_depth: bool = False, use_position: bool = True):
    """Return model_fn(seq, i, j) -> np.ndarray over split positions (shape n+1)."""

    def model_fn(seq, i, j):
        n = len(seq)
        s = torch.as_tensor(seq, dtype=torch.float32, device=DEVICE)  # 0='(' 1=')'
        sign = 1.0 - 2.0 * s                                          # '('->+1 ')'->-1

        # head A: causal counting attention -> prefix bracket depth D[p], p=0..n
        idx_p = torch.arange(n + 1, device=DEVICE).unsqueeze(1)       # (n+1,1)
        idx_q = torch.arange(n, device=DEVICE).unsqueeze(0)           # (1,n)
        causal = (idx_q < idx_p).float()                             # strict lower-tri
        D = torch.zeros(n + 1, device=DEVICE) if ablate_depth else (causal @ sign)

        # split head: a single linear QK score over candidate splits k
        Di = D[i]
        ks = torch.arange(n + 1, device=DEVICE, dtype=torch.float32)
        quad = -T_QUAD * (D - Di) * (D - Di)
        pos = BETA * (Di + 0.5 - D) * ks if use_position else torch.zeros_like(D)
        score = quad + pos

        kk = torch.arange(n + 1, device=DEVICE)
        interior = (kk > i) & (kk < j)
        masked = score.masked_fill(~interior, float("-inf"))
        out = torch.softmax(masked, dim=0)                           # 0 outside interior
        return out.detach().cpu().numpy()

    return model_fn


def canonical(payload, key):
    num = sum(r["num_cells"] for r in payload["sweep"])
    if num == 0:
        return 0.0
    return sum(r[key] * r["num_cells"] for r in payload["sweep"]) / num


def cell_type(seq, i, j, depths):
    net = depths[j] - depths[i]
    if net == -1:
        return "X (S->S R)"
    # S cell: wrapped if no interior balance point
    has_balance = any(depths[k] == depths[i] for k in range(i + 1, j))
    return "S (S->S S)" if has_balance else "S wrap (S->L X)"


def main():
    full_fn = make_model_fn()
    nopos_fn = make_model_fn(use_position=False)
    abl_fn = make_model_fn(ablate_depth=True)

    run_dir = results_dir(__file__)

    # official record: the full single-head circuit
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)

    # controls (faithfulness)
    payload_nopos = task.evaluate(nopos_fn)
    payload_abl = task.evaluate(abl_fn)

    summary = {
        "full": canonical(payload, "split_accuracy"),
        "no_position": canonical(payload_nopos, "split_accuracy"),
        "depth_ablated": canonical(payload_abl, "split_accuracy"),
        "uniform": canonical(payload, "uniform_baseline"),
    }
    per_span = {
        "span": [r["span_len"] for r in payload["sweep"]],
        "full": [r["split_accuracy"] for r in payload["sweep"]],
        "uniform": [r["uniform_baseline"] for r in payload["sweep"]],
    }

    # diverse example cells for the demo (one of each type when available)
    sym = {0: "(", 1: ")"}
    batch = task.generate(0)
    examples = []
    seen_types = set()
    for seq in batch.strings:
        n = len(seq)
        depths = [0] * (n + 1)
        for p in range(n):
            depths[p + 1] = depths[p] + (1 if seq[p] == 0 else -1)
        chart = task._cyk(seq)
        for i in range(n):
            for j in range(i + 3, n + 1):
                if not chart[i][j]:
                    continue
                ctype = cell_type(seq, i, j, depths)
                # prefer covering all three types first, then fill up to 12
                if ctype in seen_types and len(examples) >= 6:
                    continue
                seen_types.add(ctype)
                correct = sorted(task._correct_splits(chart, i, j))
                scores = full_fn(seq, i, j)
                cands = list(range(i + 1, j))
                examples.append(
                    {
                        "label": f"{''.join(sym[c] for c in seq)} cell({i},{j}) · {ctype}",
                        "seq_str": "".join(sym[c] for c in seq),
                        "cell_type": ctype,
                        "i": i,
                        "j": j,
                        "candidates": cands,
                        "scores": [float(scores[k]) for k in cands],
                        "correct": correct,
                        "depths": depths,
                    }
                )
                if len(examples) >= 12:
                    break
            if len(examples) >= 12:
                break
        if len(examples) >= 12:
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
