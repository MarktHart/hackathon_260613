"""
Attention Identity Copy — pass_5  (hand_built, with causal ablation)

Approach (delta from experiments/base_model.py):
  * ONE attention layer, NO MLP.
  * Replace RoPE with an *absolute one-hot positional* subspace concatenated
    to the token embedding in the residual stream:  x = [ tok(64) | pos_onehot(16) ].
  * Real attention: q = x @ W_Q,  k = x @ W_K,  v = x @ W_V, then
    attn = softmax(q @ k^T).  The Q/K weights are hand-set to *read only the
    positional subspace* with a per-head gain, so the score for (i, j) is
    gain_h * (pos_i . pos_j) = gain_h * delta_ij  ->  softmax peaks on the
    diagonal.  Head 0 has the largest gain (sharpest diagonal == the identity
    copier); head 7 has gain 0 (uniform attention == built-in strawman).

Why the positional subspace matters: in the canonical sweep EVERY position is
the SAME token, so token content carries zero positional signal — the only way
to attend i->i is via the positional subspace.  That makes the causal ablation
crisp: zero the positional dims of the residual stream and the diagonal (and
the copy) must collapse to the uniform-attention baseline.

Everything below runs on CUDA (real matmuls + softmax on the GPU).
"""
from agentic.experiments import load_task, record_benchmark, results_dir
import json
import numpy as np
import torch

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback

task = load_task(__file__)
Batch = task.Batch
ModelOutput = task.ModelOutput

# Canonical dims (must match task.py)
B, L, H, D = 32, 16, 8, 64
VOCAB = 256
D_TOK = 64                 # token-embedding width
D_MODEL = D_TOK + L        # residual stream = [tok | one-hot position]
D_QK = L                   # Q/K read the L-dim positional subspace

# --- fixed (hand-set / deterministic) parameters --------------------------------
_g = torch.Generator(device=DEVICE).manual_seed(0)

# Token embedding table (content). Irrelevant to the *copy* — diagonal routing
# copies whatever value sits at position i — but we keep it real & deterministic.
TOK_EMB = torch.randn(VOCAB, D_TOK, generator=_g, device=DEVICE)

# Per-head diagonal logit (== the q.k score on the diagonal). Head 0 sharpest;
# head 7 == 0 -> uniform attention -> baseline (built-in strawman head).
HEAD_LOGIT = torch.tensor(
    [12.0, 5.0, 3.5, 2.3, 1.4, 0.7, 0.25, 0.0], device=DEVICE, dtype=torch.float32
)

# P selects the positional subspace (last L dims of the residual stream).
_P = torch.zeros(D_MODEL, D_QK, device=DEVICE)
_P[D_TOK:, :] = torch.eye(L, device=DEVICE)

# W_Q[h] = sqrt(gain_h) * P,  W_K[h] = sqrt(gain_h) * P  ->  q.k = gain_h * delta_ij
_sqrt_gain = torch.sqrt(HEAD_LOGIT).view(H, 1, 1)
W_Q = _sqrt_gain * _P.unsqueeze(0)          # (H, D_MODEL, D_QK)
W_K = _sqrt_gain * _P.unsqueeze(0)          # (H, D_MODEL, D_QK)
# Value projection (per head). It reads ONLY the positional subspace, so the
# value at position i is a distinct, ~i.i.d. random vector v_i (the position's
# content). This is what makes copying NON-trivial: in the sweep every position
# holds the SAME token, so a token-derived value would be identical at every
# position and *any* attention pattern would "copy" it. By making the value the
# position's content, a diagonal head copies v_i exactly (fidelity 1) while a
# uniform head averages L distinct vectors -> cosine ~ 1/sqrt(L) = 0.25 (exactly
# the benchmark's linear baseline). The token block of W_V is therefore zero.
W_V = torch.randn(H, D_MODEL, D, generator=_g, device=DEVICE)  # (H, D_MODEL, D)
W_V[:, :D_TOK, :] = 0.0                                          # values read position only

POS_ONEHOT = torch.eye(L, device=DEVICE)    # (L, L)


def _make_model_fn(ablate_routing: bool):
    """Return a model_fn(batch)->ModelOutput.

    If ablate_routing, the head's Q/K positional READ is knocked out (scores -> 0
    => uniform attention) while the VALUES are left completely intact. This is the
    clean causal test: same value content, but the routing that produced the
    diagonal is removed. Copy fidelity must then fall to the uniform baseline."""
    def model_fn(batch: Batch) -> ModelOutput:
        tokens = torch.as_tensor(batch.tokens, dtype=torch.long, device=DEVICE)  # (B,L)
        Bb, Ll = tokens.shape
        tok = TOK_EMB[tokens]                                    # (B,L,D_TOK)
        pos = POS_ONEHOT.unsqueeze(0).expand(Bb, Ll, L)          # (B,L,L)
        x = torch.cat([tok, pos], dim=-1)                        # (B,L,D_MODEL)

        if ablate_routing:
            # Knock out the diagonal-producing routing; values are unchanged.
            scores = torch.zeros(Bb, H, Ll, Ll, device=DEVICE)
        else:
            q = torch.einsum("bld,hdk->bhlk", x, W_Q)           # (B,H,L,D_QK)
            k = torch.einsum("bld,hdk->bhlk", x, W_K)           # (B,H,L,D_QK)
            scores = torch.einsum("bhik,bhjk->bhij", q, k)      # (B,H,L,L)
        attn = torch.softmax(scores, dim=-1)                     # (B,H,L,L)
        v = torch.einsum("bld,hde->bhle", x, W_V)               # (B,H,L,D)

        return ModelOutput(
            attn_weights=attn.detach().cpu().numpy().astype(np.float32),
            values=v.detach().cpu().numpy().astype(np.float32),
        )
    return model_fn


def _per_head(model_fn, token: int):
    """Per-head copy fidelity + diagonal mass at a single sweep token (mirrors
    task.evaluate's formula but keeps ALL heads, for the leaderboard chart)."""
    tokens = np.full((B, L), token, dtype=np.int32)
    out = model_fn(Batch(tokens=tokens))
    attn, vals = out.attn_weights, out.values
    attn_out = np.einsum("bhij,bhjd->bhid", attn, vals)
    eps = 1e-8
    ao = attn_out / (np.linalg.norm(attn_out, axis=-1, keepdims=True) + eps)
    vn = vals / (np.linalg.norm(vals, axis=-1, keepdims=True) + eps)
    cos = np.sum(ao * vn, axis=-1)                       # (B,H,L)
    fid = cos.mean(axis=(0, 2))                          # (H,)
    diag = attn[:, :, np.arange(L), np.arange(L)].mean(axis=(0, 2))  # (H,)
    head0_attn = attn[0, 0].tolist()                    # (L,L) sample for heatmap
    return fid.tolist(), diag.tolist(), head0_attn


def main():
    real_fn = _make_model_fn(ablate_routing=False)
    abl_fn = _make_model_fn(ablate_routing=True)

    # Headline payload (scored by benchmark.py) uses the REAL head.
    payload = task.evaluate(real_fn)
    abl_payload = task.evaluate(abl_fn)

    run_dir = results_dir(__file__)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Per-head breakdown + diagonal heatmap at the canonical token.
    fid, diag, head0_attn = _per_head(real_fn, task.CANONICAL_TOKEN)

    sweep_real = [
        {"token": r["token"], "fidelity": r["copy_fidelity"],
         "diag": r["diag_attn_mass"], "best_head": r["best_head"]}
        for r in payload["sweep"]
    ]
    sweep_ablated = [
        {"token": r["token"], "fidelity": r["copy_fidelity"], "diag": r["diag_attn_mass"]}
        for r in abl_payload["sweep"]
    ]

    artifacts = {
        "canonical_token": task.CANONICAL_TOKEN,
        "config": payload["config"],
        "per_head": {
            "head": list(range(H)),
            "fidelity": fid,
            "diag_mass": diag,
            "head_logit": HEAD_LOGIT.detach().cpu().tolist(),
        },
        "sweep_real": sweep_real,
        "sweep_ablated": sweep_ablated,
        "head0_attn": head0_attn,
    }
    with (run_dir / "artifacts.json").open("w") as f:
        json.dump(artifacts, f, indent=2)
    with (run_dir / "payload.json").open("w") as f:
        json.dump(payload, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    can = next(r for r in sweep_real if r["token"] == task.CANONICAL_TOKEN)
    print(f"[pass_5] canonical fidelity={can['fidelity']:.4f} "
          f"best_head={can['best_head']} "
          f"(ablated={sweep_ablated[2]['fidelity']:.4f})")
    print(f"benchmark recorded to {run_dir / 'benchmark.json'}")


if __name__ == "__main__":
    main()
