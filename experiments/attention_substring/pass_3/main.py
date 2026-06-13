"""
attention_substring / pass_3 — hand_built two-layer induction circuit.

A faithful, position-general substring-matching circuit, written out by hand as
torch weights on CUDA (no training, no label leakage). It is a minimal delta from
`base_model.py`: two single-head attention layers (no MLP, no unembed used).

The task asks: at the position right after the *second* occurrence of a repeated
pattern (`target_pos`), does a head attend back to the last token of the *first*
occurrence (`source_pos`)?  Note that token[target_pos-1] == token[source_pos]
== L (the pattern's last token).  So the circuit must:

  Layer 0  — previous-token head.  Every position p attends to p-1 (built from
             explicit position one-hots) and copies token[p-1] into a dedicated
             "prev" subspace of the residual stream.
  Layer 1  — matching head.  The query at position q reads its prev subspace
             (= token[q-1]); the key at position k reads its current token
             (= token[k]).  The score is high wherever token[q-1] == token[k].
             A monotone "earliest-wins" key bias breaks ties toward the FIRST
             occurrence, so at q=target_pos the argmax lands on source_pos.

Crucially the weights never read source_pos / target_pos — the same matrices run
on every position.  Ablating layer 0 (zeroing the prev-token write) collapses the
match to chance, which is our causal-faithfulness check.
"""

import json
import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

SEQ_LEN = 64
VOCAB = 64
D = 192                       # [0:64]=token one-hot, [64:128]=prev one-hot, [128:192]=pos one-hot
TOK = slice(0, 64)
PREV = slice(64, 128)
POS = slice(128, 192)
SCALE0 = 50.0                 # sharpen layer-0 prev-token attention
SCALE1 = 50.0                 # sharpen layer-1 content match
GAMMA = 4.0                   # earliest-position key bias slope
NEG = -1.0e9                  # causal mask fill


# ---------------------------------------------------------------------------
# Hand-set constant weights (all on GPU).
# ---------------------------------------------------------------------------
def _build_weights():
    WE = torch.zeros(VOCAB, D, device=DEVICE)            # token t -> one_hot(t) in TOK block
    idx = torch.arange(VOCAB, device=DEVICE)
    WE[idx, idx] = 1.0

    POSE = torch.zeros(SEQ_LEN, D, device=DEVICE)        # position p -> one_hot(p) in POS block
    p = torch.arange(SEQ_LEN, device=DEVICE)
    POSE[p, 128 + p] = 1.0

    # Layer-0 key projection: maps POS one-hot(k) -> one_hot(k+1) (a +1 shift),
    # so q0(p)=one_hot(p) matches k0(k) iff p == k+1  <=>  k = p-1.
    K0mat = torch.zeros(D, 64, device=DEVICE)
    i = torch.arange(63, device=DEVICE)
    K0mat[128 + i, i + 1] = 1.0

    # Causal mask (allow k <= q).
    cm = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN, device=DEVICE), diagonal=1) * NEG

    # Earliest-wins key bias: -GAMMA * k, favouring the first matching key.
    keybias = -GAMMA * torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.float32)

    return WE, POSE, K0mat, cm, keybias


_WE, _POSE, _K0mat, _CM, _KEYBIAS = _build_weights()


@torch.no_grad()
def _forward(tokens_np: np.ndarray, ablate_prev: bool = False) -> np.ndarray:
    """tokens_np: [seq_len] int.  Returns attn [n_layers=2, n_heads=1, L, L] float32."""
    tokens = torch.as_tensor(tokens_np, dtype=torch.long, device=DEVICE)
    x = _WE[tokens] + _POSE                              # [L, D]

    # ---- Layer 0: previous-token head ----
    q0 = x[:, POS]                                       # one_hot(p)
    k0 = x @ _K0mat                                      # one_hot(k+1)
    s0 = SCALE0 * (q0 @ k0.t()) + _CM
    a0 = F.softmax(s0, dim=-1)                           # peaks at k = p-1
    v0 = x[:, TOK]                                       # token[k] one-hot
    prev_feat = a0 @ v0                                  # [L, 64] == token[p-1] one-hot

    x1 = x.clone()
    if not ablate_prev:
        x1[:, PREV] = prev_feat                          # write token[p-1] into prev subspace

    # ---- Layer 1: content-matching (induction) head ----
    q1 = x1[:, PREV]                                     # token[q-1] one-hot
    k1 = x1[:, TOK]                                      # token[k]  one-hot
    match = q1 @ k1.t()                                  # 1 where token[q-1]==token[k]
    s1 = SCALE1 * match + _KEYBIAS[None, :] + _CM
    a1 = F.softmax(s1, dim=-1)

    attn = torch.stack([a0, a1], dim=0).unsqueeze(1)     # [2, 1, L, L]
    return attn.detach().cpu().numpy().astype(np.float32)


def make_model_fn(ablate_prev: bool = False):
    def model_fn(input_ids: np.ndarray) -> dict:
        attn = _forward(input_ids[0], ablate_prev=ablate_prev)
        return {"attn_weights": attn}      # logits intentionally omitted
    return model_fn


# ---------------------------------------------------------------------------
def _detection(payload: dict) -> float:
    sweep = payload["sweep"]
    return float(np.mean([1.0 if r["correct_top1"] else 0.0 for r in sweep]))


def _cell_detection(payload: dict) -> dict:
    out = {}
    for L in (2, 3, 4):
        for Dd in (8, 16, 32):
            cell = [1.0 if r["correct_top1"] else 0.0
                    for r in payload["sweep"]
                    if r["pattern_length"] == L and r["distance"] == Dd]
            out[f"plen{L}_dist{Dd}"] = float(np.mean(cell)) if cell else 0.0
    return out


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # --- Induction circuit (the official scored result) ---
    payload = task.evaluate(make_model_fn(ablate_prev=False))
    record_benchmark(__file__, run_dir, payload)
    det_ind = _detection(payload)
    cells = _cell_detection(payload)

    # --- Faithfulness ablation: knock out the layer-0 prev-token head ---
    payload_abl = task.evaluate(make_model_fn(ablate_prev=True))
    det_abl = _detection(payload_abl)

    random_baseline = 1.0 / (SEQ_LEN - 1)

    comparison = {
        "induction_detection": det_ind,
        "ablation_detection": det_abl,
        "random_baseline": random_baseline,
        "cells": cells,
    }
    with open(run_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # --- A few example attention rows for the Demo heatmap/bar ---
    batch = task.generate(seed=42)
    examples = []
    wanted = [(2, 8), (3, 16), (4, 32)]
    for (L, Dd) in wanted:
        hit = np.where((batch.pattern_lengths == L) & (batch.distances == Dd))[0]
        if len(hit) == 0:
            continue
        i = int(hit[0])
        attn = _forward(batch.input_ids[i])           # [2,1,L,L]
        tpos = int(batch.target_positions[i])
        spos = int(batch.source_positions[i])
        row = attn[1, 0, tpos, :].tolist()            # layer-1 attention from target_pos
        examples.append({
            "label": f"plen={L}, dist={Dd}",
            "target_pos": tpos,
            "source_pos": spos,
            "attn": row,
            "tokens": [int(t) for t in batch.input_ids[i].tolist()],
        })
    with open(run_dir / "examples.json", "w") as f:
        json.dump(examples, f, indent=2)

    print(f"[pass_3] induction detection = {det_ind:.3f}  "
          f"(ablation = {det_abl:.3f}, random = {random_baseline:.4f})")
    print(f"[pass_3] per-cell detection: {cells}")
    print(f"[pass_3] benchmark + artefacts written to {run_dir}")


if __name__ == "__main__":
    main()
