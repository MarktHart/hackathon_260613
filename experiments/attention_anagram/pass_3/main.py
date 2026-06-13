"""attention_anagram / pass_3 — a HAND-BUILT token-identity matching head.

Delta from base_model.py: one cross-attention layer (no MLP, no RoPE/positional
term). The query reads the TARGET token embedding, the key reads the SOURCE token
embedding, and the QK circuit is hand-set to the identity over the vocabulary:

    E   = I_vocab            (one-hot token embedding, d_model = vocab)
    W_Q = W_K = I            (identity projections)
    scores[t, s] = beta * <E[tgt_t], E[src_s]>  =  beta if tgt_t == src_s else 0
    attn = softmax_s(scores)

No training. The mechanism is written out by hand: each target token attends the
source position(s) holding the SAME token id. For tokens that are unique in the
source (the common case at vocab=50, L=8) this puts ~all attention on the true
source position; when a token repeats, attention splits equally over its copies,
so alignment on the *true* position is 1/(copies). This is the only honest
failure mode and we sweep it explicitly (the vocab axis).

Three model functions are evaluated under identical conditions so the claim is
testable, not asserted:
  * match      — the hand-built token-identity circuit (the mechanism)
  * positional — strawman: attend source pos == target pos, ignoring tokens
  * ablated    — knock out the QK circuit (scores->0): uniform attention

Everything runs on CUDA.
"""
import argparse
import json
from math import comb

import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback.
VOCAB = 50
SEQ = 8
N_HEADS = 8
BETA = 30.0  # softmax temperature of the hand-set circuit (large => hard match)


# --------------------------------------------------------------------------
# Hand-built circuit + controls, all expressed as torch ops on CUDA.
# --------------------------------------------------------------------------
def _attn_from_scores(scores):
    """scores: (B, L, L) -> (B, N_HEADS, L, L) softmaxed over source dim."""
    attn = torch.softmax(scores, dim=-1)
    return attn.unsqueeze(1).expand(-1, N_HEADS, -1, -1).contiguous()


def make_match_fn(beta=BETA, vocab=VOCAB):
    """Hand-set token-identity matcher. E=I, W_Q=W_K=I => scores = beta*(tok match)."""
    def _fn(src_ids, tgt_ids):
        with torch.no_grad():
            src = torch.as_tensor(np.asarray(src_ids), dtype=torch.long, device=DEVICE)
            tgt = torch.as_tensor(np.asarray(tgt_ids), dtype=torch.long, device=DEVICE)
            qoh = F.one_hot(tgt, vocab).float()             # (B, L, V)  query = E[tgt]
            koh = F.one_hot(src, vocab).float()             # (B, L, V)  key   = E[src]
            scores = beta * (qoh @ koh.transpose(-1, -2))   # (B, Lt, Ls); 1 where ids match
            attn = _attn_from_scores(scores)
        return attn.detach().cpu().numpy().astype(np.float32)
    return _fn


def make_positional_fn(beta=BETA):
    """Strawman: each target position attends the SAME source position (ignores tokens)."""
    def _fn(src_ids, tgt_ids):
        with torch.no_grad():
            src = torch.as_tensor(np.asarray(src_ids), dtype=torch.long, device=DEVICE)
            B, L = src.shape
            scores = beta * torch.eye(L, device=DEVICE).unsqueeze(0).expand(B, -1, -1)
            attn = _attn_from_scores(scores)
        return attn.detach().cpu().numpy().astype(np.float32)
    return _fn


def make_ablated_fn():
    """Faithfulness control: knock the QK circuit out (scores->0) => uniform attention."""
    def _fn(src_ids, tgt_ids):
        with torch.no_grad():
            src = torch.as_tensor(np.asarray(src_ids), dtype=torch.long, device=DEVICE)
            B, L = src.shape
            scores = torch.zeros(B, L, L, device=DEVICE)
            attn = _attn_from_scores(scores)
        return attn.detach().cpu().numpy().astype(np.float32)
    return _fn


# --------------------------------------------------------------------------
# Operating-range helpers (own random anagrams at varying L / vocab).
# --------------------------------------------------------------------------
def _random_anagrams(rng, B, L, vocab):
    src = rng.integers(0, vocab, size=(B, L)).astype(np.int64)
    perm = np.stack([rng.permutation(L) for _ in range(B)])
    tgt = np.take_along_axis(src, perm, axis=1)
    return src, tgt, perm


def _mean_alignment(model_fn, src, tgt, perm):
    """Mean attention (over heads, positions, batch) on the TRUE source position."""
    attn = model_fn(src, tgt)                      # (B, H, L, L)
    attn = attn.mean(1)                            # mean over heads -> (B, L, L)
    B, L, _ = attn.shape
    idx = np.arange(B)
    vals = [attn[idx, t, perm[:, t]].mean() for t in range(L)]
    return float(np.mean(vals))


def sweep_seq_len(rng):
    """Vary L over 2 orders of magnitude. Vocab is scaled with L (vocab = 16*L) so
    the per-token collision rate stays fixed -> this isolates the claim that the
    circuit has NO positional dependence (alignment is flat in L), separate from
    the vocab/collision axis swept below."""
    seq_lens = [2, 4, 8, 16, 32, 64, 128, 256]
    out = []
    for L in seq_lens:
        V = 16 * L
        fn = make_match_fn(vocab=V)
        src, tgt, perm = _random_anagrams(rng, 200, L, V)
        out.append(_mean_alignment(fn, src, tgt, perm))
    return seq_lens, out, [1.0 / L for L in seq_lens]


def sweep_vocab(rng, L=SEQ):
    vocabs = [8, 12, 16, 24, 32, 50, 100, 200, 400]
    out, expected = [], []
    for V in vocabs:
        fn = make_match_fn(vocab=V)
        src, tgt, perm = _random_anagrams(rng, 400, L, V)
        out.append(_mean_alignment(fn, src, tgt, perm))
        # Analytic expectation given collisions: E[1 / (1 + #other src copies)].
        p = 1.0 / V
        e = 0.0
        for k in range(L):  # k of the other L-1 positions also hold this token
            pk = comb(L - 1, k) * (p ** k) * ((1 - p) ** (L - 1 - k))
            e += pk / (k + 1)
        expected.append(e)
    return vocabs, out, expected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=BETA)
    args = ap.parse_args()

    task = load_task(__file__)

    # Headline payload = the hand-built token-identity matcher.
    match_fn = make_match_fn(beta=args.beta)
    payload = task.evaluate(match_fn)

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    with open(run_dir / "payload.json", "w") as f:
        json.dump(payload, f)

    # Controls under identical conditions.
    pos_payload = task.evaluate(make_positional_fn(beta=args.beta))
    abl_payload = task.evaluate(make_ablated_fn())

    def per_perm(p):
        return {
            r["perm_type"]: float(np.mean([h["mean_alignment"] for h in r["head_alignments"]]))
            for r in p["sweep"]
        }

    rng = np.random.default_rng(123)
    seq_lens, seq_align, seq_base = sweep_seq_len(rng)
    vocabs, voc_align, voc_expected = sweep_vocab(rng)

    # Hand-set QK matrix over the vocabulary (the mechanism) = beta * Identity.
    qk = (args.beta * np.eye(VOCAB)).astype(np.float32)

    diag = {
        "beta": args.beta,
        "uniform_baseline": 1.0 / SEQ,
        "perm_bars": {
            "match": per_perm(payload),
            "positional": per_perm(pos_payload),
            "ablated": per_perm(abl_payload),
        },
        "canonical": {
            "match": per_perm(payload).get("random", 0.0),
            "positional": per_perm(pos_payload).get("random", 0.0),
            "ablated": per_perm(abl_payload).get("random", 0.0),
        },
        "qk_matrix": qk.tolist(),
        "op_seq": {"seq_lens": seq_lens, "alignment": seq_align, "baseline": seq_base},
        "op_vocab": {"vocabs": vocabs, "alignment": voc_align, "expected": voc_expected},
    }
    with open(run_dir / "diag.json", "w") as f:
        json.dump(diag, f)

    print("canonical (random perm) alignment:")
    print("  match circuit  :", diag["canonical"]["match"])
    print("  positional     :", diag["canonical"]["positional"])
    print("  ablated (QK->0):", diag["canonical"]["ablated"])
    print("  uniform base   :", diag["uniform_baseline"])
    print("op seq_len align :", seq_align)
    print("op vocab  align  :", voc_align)


if __name__ == "__main__":
    main()
