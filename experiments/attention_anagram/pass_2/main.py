"""attention_anagram / pass_2 — a *trained* single-layer attention head learns
token-matching on its own.

Delta from base_model.py: one attention-only layer (no MLP, no positional
embedding). Query = W_Q . Emb[target], Key = W_K . Emb[source], 8 heads. We
TRAIN it (cross-entropy of each head's attention against the true source
position) on fresh random anagrams. The interp claim: with no hand-set weights,
the head converges to a token-identity matcher — its effective QK matrix
M[a,b] = <W_Q e_a, W_K e_b> becomes diagonal (a target token attends the source
token of the SAME id). Because there is no positional component, the learned
circuit transfers to any sequence length -> operating-range sweep.

Everything runs on CUDA.
"""
import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback.
VOCAB = 50
SEQ = 8
N_HEADS = 8
D_MODEL = 64


class AnagramAttn(nn.Module):
    def __init__(self, vocab=VOCAB, d=D_MODEL, n_heads=N_HEADS):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.Wq = nn.Linear(d, d, bias=False)
        self.Wk = nn.Linear(d, d, bias=False)
        self.n_heads = n_heads
        self.hd = d // n_heads

    def forward(self, src, tgt):
        B, L = src.shape
        es, et = self.tok(src), self.tok(tgt)
        q = self.Wq(et).view(B, L, self.n_heads, self.hd).transpose(1, 2)
        k = self.Wk(es).view(B, L, self.n_heads, self.hd).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)  # B,H,Lt,Ls
        return torch.softmax(scores, dim=-1), scores

    def token_match_matrix(self):
        """Effective QK matrix over the vocab: M[a,b] = score(tgt tok a, src tok b)."""
        E = self.tok.weight                       # (V, d)
        Q, K = self.Wq(E), self.Wk(E)             # (V, d)
        return (Q @ K.t()) / math.sqrt(self.hd)   # (V, V)


def _gen(rng, B, L, vocab=VOCAB):
    src = rng.integers(0, vocab, size=(B, L))
    perm = np.zeros((B, L), dtype=np.int64)
    for b in range(B):
        t = rng.integers(0, 3)
        if t == 0:
            p = np.arange(L); i, j = rng.choice(L, 2, replace=False); p[i], p[j] = p[j], p[i]
        elif t == 1:
            p = (np.arange(L) + rng.integers(1, L)) % L
        else:
            p = rng.permutation(L)
        perm[b] = p
    tgt = np.take_along_axis(src, perm, axis=1)
    return src, tgt, perm


def train(steps=900, B=256, lr=2e-3, seed=1):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = AnagramAttn().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for s in range(steps):
        src, tgt, perm = _gen(rng, B, SEQ)
        src_t = torch.as_tensor(src, dtype=torch.long, device=DEVICE)
        tgt_t = torch.as_tensor(tgt, dtype=torch.long, device=DEVICE)
        perm_t = torch.as_tensor(perm, device=DEVICE)            # B,L true src pos
        _, scores = model(src_t, tgt_t)                          # B,H,Lt,Ls
        loss = F.cross_entropy(
            scores.reshape(-1, SEQ),
            perm_t.unsqueeze(1).expand(-1, N_HEADS, -1).reshape(-1),
        )
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def make_model_fn(model):
    model.eval()

    def model_fn(src_ids, tgt_ids):
        with torch.no_grad():
            src = torch.as_tensor(src_ids, dtype=torch.long, device=DEVICE)
            tgt = torch.as_tensor(tgt_ids, dtype=torch.long, device=DEVICE)
            attn, _ = model(src, tgt)
        return attn.detach().cpu().numpy().astype(np.float32)
    return model_fn


def _align_at_len(model, L, rng):
    """Mean attention on the true source position at sequence length L."""
    src, tgt, perm = _gen(rng, 200, L)
    src_t = torch.as_tensor(src, dtype=torch.long, device=DEVICE)
    tgt_t = torch.as_tensor(tgt, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        attn, _ = model(src_t, tgt_t)                  # 200,H,L,L
    attn = attn.mean(1).cpu().numpy()                  # mean over heads -> 200,L,L
    idx = np.arange(200)
    vals = [attn[idx, t, perm[:, t]].mean() for t in range(L)]
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    args = ap.parse_args()

    task = load_task(__file__)

    model = train(steps=args.steps)
    payload = task.evaluate(make_model_fn(model))

    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    with open(run_dir / "payload.json", "w") as f:
        json.dump(payload, f)

    rng = np.random.default_rng(123)

    # (a) Untrained strawman: same architecture, random init.
    torch.manual_seed(7)
    untrained = AnagramAttn().to(DEVICE)
    untr_payload = task.evaluate(make_model_fn(untrained))

    def canon(p):
        rec = next(r for r in p["sweep"] if r["perm_type"] == "random")
        return float(np.mean([h["mean_alignment"] for h in rec["head_alignments"]]))

    # (b) Learned token-match matrix (should be diagonal).
    M = model.token_match_matrix().detach().cpu().numpy()
    diag = float(np.mean(np.diag(M)))
    offdiag = float((M.sum() - np.trace(M)) / (VOCAB * VOCAB - VOCAB))

    # (c) Operating range over seq_len (2+ orders of magnitude).
    seq_lens = [2, 4, 8, 16, 32, 64, 128, 256]
    op_align = [_align_at_len(model, L, rng) for L in seq_lens]
    op_baseline = [1.0 / L for L in seq_lens]

    diag_out = {
        "trained_canonical": canon(payload),
        "untrained_canonical": canon(untr_payload),
        "uniform_baseline": 1.0 / SEQ,
        "perm_bars": {
            r["perm_type"]: float(np.mean([h["mean_alignment"] for h in r["head_alignments"]]))
            for r in payload["sweep"]
        },
        "untrained_perm_bars": {
            r["perm_type"]: float(np.mean([h["mean_alignment"] for h in r["head_alignments"]]))
            for r in untr_payload["sweep"]
        },
        "token_match_matrix": M.tolist(),
        "tm_diag_mean": diag,
        "tm_offdiag_mean": offdiag,
        "op_range": {"seq_lens": seq_lens, "alignment": op_align, "baseline": op_baseline},
    }
    with open(run_dir / "diag.json", "w") as f:
        json.dump(diag_out, f)

    print("trained canonical :", diag_out["trained_canonical"])
    print("untrained canonical:", diag_out["untrained_canonical"])
    print("token-match diag/offdiag:", diag, offdiag)
    print("op-range alignment:", op_align)


if __name__ == "__main__":
    main()
