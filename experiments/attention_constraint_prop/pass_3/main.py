"""
Hand-built bracket-matching attention circuit (no training).

Approach (attempt type: hand_built + interp/ablation)
-----------------------------------------------------
We construct, by hand, a single attention layer whose weights are SET, not
learned, so that each bracket token attends to its matching partner. The
circuit is a small delta from `base_model.py`: one attention layer, a 4-dim
token embedding (one-hot over the four bracket tokens), hand-set Q/K
projections that implement a bracket "partner" lookup, and a hand-set
relative-position bias (ALiBi/T5-style) that breaks ties toward the nearest
candidate partner.

Mechanism, in two pieces of QK structure:
  1. TYPE MATCH (the circuit we claim).  Token feature f(t) is a one-hot over
     {OPEN_A, CLOSE_A, OPEN_B, CLOSE_B} (filler -> zero vector).  W_Q = M, the
     4x4 partner-permutation matrix (OPEN_A<->CLOSE_A, OPEN_B<->CLOSE_B); W_K = I.
     Then  Q_i . K_j = f_i^T M f_j  is 1 exactly when j is a *partner-type*
     token of i, else 0.  Scaled by BIG this dominates the softmax, so an
     OPEN_A query spreads weight over CLOSE_A keys (and vice-versa) and ignores
     filler / same-type tokens entirely.
  2. PROXIMITY (a relative-position prior, varied per head).  We add
     `-alpha * |i - j|` to the scores.  With BIG >> alpha*max_dist this never
     lets a non-partner win; it only sharpens attention onto the *nearest*
     partner-type token.  alpha=0 -> uniform over all partner-type tokens;
     larger alpha -> committed to the nearest one.  Sweeping alpha across heads
     is exactly the distance/fidelity trade-off the goal asks about.

Faithfulness (causal evidence).  Because the weights are hand-set we can ABLATE
the bracket-matching term directly (type_scale -> 0).  We evaluate that ablated
circuit and a positional-only strawman under identical conditions; both collapse
to ~random, while the full circuit is many-x random.  This is the causal claim
the metric alone cannot make.

Everything runs in torch on CUDA.
"""

import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; never fall back to CPU

# Bracket vocabulary (mirrors task.py)
OPEN_A, CLOSE_A, OPEN_B, CLOSE_B = 100, 101, 102, 103
VOCAB_SIZE = 104

# Per-head proximity strengths. alpha=0 -> uniform over partner-type tokens;
# larger alpha -> commit to the nearest partner. The benchmark takes the best
# head, so this sweep also visualises the distance/fidelity trade-off.
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5)

# BIG must dominate alpha*max_dist so a non-partner key can never outscore a
# real partner. max alpha=1.5, max |i-j| ~ 127 (largest seq_len we probe) ->
# alpha*dist <= ~190; BIG=400 keeps partners strictly on top.
BIG = 400.0


def _partner_matrix(device) -> torch.Tensor:
    """4x4 permutation: OPEN_A<->CLOSE_A (0<->1), OPEN_B<->CLOSE_B (2<->3)."""
    M = torch.zeros(4, 4, device=device)
    M[0, 1] = M[1, 0] = 1.0
    M[2, 3] = M[3, 2] = 1.0
    return M


def _token_features(ids: torch.Tensor) -> torch.Tensor:
    """[B,S] long -> [B,S,4] one-hot bracket features (filler -> all zeros)."""
    feat = torch.zeros(*ids.shape, 4, device=ids.device)
    feat[..., 0] = (ids == OPEN_A).float()
    feat[..., 1] = (ids == CLOSE_A).float()
    feat[..., 2] = (ids == OPEN_B).float()
    feat[..., 3] = (ids == CLOSE_B).float()
    return feat


def circuit_attention(
    input_ids: np.ndarray,
    alphas=ALPHAS,
    type_scale: float = 1.0,
    mask_self: bool = False,
) -> torch.Tensor:
    """Hand-built attention. Returns [B, L=1, H=len(alphas), S, S] on DEVICE.

    type_scale=1.0  -> full circuit.
    type_scale=0.0  -> ABLATED (bracket matching removed; positional only).
    mask_self=True  -> diagonal removed (positional-only strawman attends to a
                       neighbour, never itself).
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long, device=DEVICE)
    B, S = ids.shape

    F = _token_features(ids)                 # [B,S,4]
    M = _partner_matrix(DEVICE)              # W_Q = M
    Q = F @ M                                # [B,S,4]  partner indicator of query
    K = F                                    # W_K = I
    type_match = torch.einsum("bsd,btd->bst", Q, K)  # [B,S,S] in {0,1}

    pos = torch.arange(S, device=DEVICE).float()
    dist = (pos[:, None] - pos[None, :]).abs()       # [S,S]

    if mask_self:
        self_pen = torch.eye(S, device=DEVICE)[None] * 1e4
    else:
        self_pen = 0.0

    heads = []
    for alpha in alphas:
        scores = (BIG * type_scale) * type_match - alpha * dist[None] - self_pen
        heads.append(torch.softmax(scores, dim=-1))
    A = torch.stack(heads, dim=1)            # [B,H,S,S]
    return A.unsqueeze(1)                    # [B,1,H,S,S]


def make_model_fn(alphas=ALPHAS, type_scale: float = 1.0, mask_self: bool = False):
    """Wrap the circuit as a task.evaluate-compatible model_fn (NumPy in/out)."""

    def model_fn(input_ids: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            A = circuit_attention(input_ids, alphas, type_scale, mask_self)
            return A.detach().cpu().numpy().astype(np.float32)

    return model_fn


def _fidelity(payload: dict) -> float:
    """max-head alignment at the canonical distance / uniform baseline."""
    cd = payload["config"]["canonical_distance"]
    sl = payload["config"]["seq_len"]
    baseline = 1.0 / sl
    for rec in payload["sweep"]:
        if rec["distance"] == cd:
            return rec["max_alignment"] / baseline
    return 0.0


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # ---- Headline: full hand-built circuit on the canonical batch ----------
    full_fn = make_model_fn(type_scale=1.0)
    payload = task.evaluate(full_fn)
    record_benchmark(__file__, run_dir, payload)

    baseline = 1.0 / payload["config"]["seq_len"]
    fid_full = _fidelity(payload)
    print(f"[full]      constraint_propagation_fidelity = {fid_full:.3f}x  "
          f"(best head = {payload['sweep'][[r['distance'] for r in payload['sweep']].index(payload['config']['canonical_distance'])]['best_head']})")

    # ---- Causal evidence: ablate the bracket-matching term -----------------
    abl_fn = make_model_fn(type_scale=0.0)                  # no type match
    fid_abl = _fidelity(task.evaluate(abl_fn))
    straw_fn = make_model_fn(type_scale=0.0, mask_self=True)  # nearest-neighbour
    fid_straw = _fidelity(task.evaluate(straw_fn))
    fid_uniform = _fidelity(task.evaluate(task.random_model_fn()))
    print(f"[ablated]   fidelity = {fid_abl:.3f}x   "
          f"[strawman/nn] = {fid_straw:.3f}x   [uniform] = {fid_uniform:.3f}x")

    # ---- Per-head sweep (alignment vs distance, per alpha) ------------------
    per_head_sweep = []
    for rec in payload["sweep"]:
        row = {"distance": rec["distance"], "n_entries": rec["n_entries"],
               "heads": [h["alignment"] for h in rec["heads"]]}  # indexed by alpha
        per_head_sweep.append(row)

    # ---- Operating range: vary seq_len (baseline=1/seq_len shifts) ----------
    seq_len_sweep = []
    for sl in (16, 32, 64, 128):
        batch = task.generate(seed=0, num_sequences=200, seq_len=sl)
        p = task.evaluate(full_fn, batch=batch)
        seq_len_sweep.append({"seq_len": sl, "baseline": 1.0 / sl,
                              "fidelity": _fidelity(p)})
        print(f"[range] seq_len={sl:4d}  fidelity={seq_len_sweep[-1]['fidelity']:.3f}x")

    # ---- A few example sequences (with a canonical-distance pair) for viz ---
    cbatch = task.generate(seed=0)
    cd = payload["config"]["canonical_distance"]
    chosen = []
    for b, entries in enumerate(cbatch.constraints):
        if any(d == cd for (_, _, d) in entries):
            chosen.append(b)
        if len(chosen) >= 6:
            break
    ex_ids = cbatch.input_ids[chosen]                      # [E,S]
    ex_attn = full_fn(ex_ids)[:, 0]                        # [E,H,S,S]
    examples = [{"tokens": cbatch.input_ids[b].tolist(),
                 "pairs": [[int(i), int(j), int(d)] for (i, j, d) in cbatch.constraints[b]]}
                for b in chosen]

    np.savez(f"{run_dir}/examples.npz",
             input_ids=ex_ids.astype(np.int64),
             attn=ex_attn.astype(np.float32))

    analysis = {
        "alphas": list(ALPHAS),
        "baseline": baseline,
        "fidelity_full": fid_full,
        "fidelity_ablated": fid_abl,
        "fidelity_strawman_nn": fid_straw,
        "fidelity_uniform": fid_uniform,
        "per_head_sweep": per_head_sweep,
        "seq_len_sweep": seq_len_sweep,
        "examples": examples,
        "model_info": payload["model_info"],
        "canonical_distance": cd,
    }
    with open(f"{run_dir}/analysis.json", "w") as fh:
        json.dump(analysis, fh, indent=2)

    print(f"Wrote artefacts to {run_dir}")


if __name__ == "__main__":
    main()
