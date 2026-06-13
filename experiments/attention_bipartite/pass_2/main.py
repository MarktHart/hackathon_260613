"""attention_bipartite / pass_2 — hand-built bipartite-mask attention.

WHY THIS FILE LOOKS DIFFERENT FROM THE HARNESS SNIPPET
------------------------------------------------------
The shipped ``task.evaluate`` cannot be used as-is: it has two bugs.
  1. It calls ``np.softmax`` which does not exist in NumPy -> AttributeError.
  2. It computes the reported attention statistics from the *raw* q/k it was
     handed and never looks at ``model_fn``'s output, so no attempt's mechanism
     can ever change the score.
We still drive everything off the canonical generator ``task.generate(seed=42)``
(so the data is byte-identical to every other attempt), but we re-implement the
*documented payload contract* from the goal README faithfully: the reported
``mean_attn_within`` / ``mean_attn_between`` / ``retrieval_acc`` are derived from
the attention weights our model actually produces. All compute runs in torch on
CUDA.

THE MECHANISM (delta from base_model.py)
----------------------------------------
The token vectors carry *content only*: q=k=v=feature_base[fid]+noise. They hold
NO position/group information. So standard content attention attends to the
self/within-group token (highest dot product) and FAILS the bipartite task.
To attend *between* groups you must inject the group structure. Our delta from
``base_model.Attention`` is a single fixed additive mask applied to the scores
before softmax (exactly like the causal mask there, but bipartite): score(i,j)
is set to -inf when i and j are in the SAME group. That is the whole circuit;
ablating it (mask off) recovers the failing baseline.
"""
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback
NUM_HEADS_SWEEP = [1, 2, 4, 8]

task = load_task(__file__)


# --------------------------------------------------------------------------- #
# Core hand-built circuit (torch / CUDA)
# --------------------------------------------------------------------------- #
def within_group_mask(seq_len: int, group_size: int, device) -> torch.Tensor:
    """(S, S) bool, True where query i and key j are in the SAME group."""
    idx = torch.arange(seq_len, device=device)
    group = (idx >= group_size).long()          # 0 = Group A, 1 = Group B
    return group[:, None] == group[None, :]


def attention(q, k, v, group_size, bipartite: bool):
    """Scaled dot-product attention on (N, S, hd) CUDA tensors.

    bipartite=True  -> add the fixed cross-group mask (the proposed circuit).
    bipartite=False -> plain content attention (the strawman baseline).
    Returns (out, attn_weights).
    """
    N, S, hd = q.shape
    scale = 1.0 / (hd ** 0.5)
    scores = torch.einsum("nqd,nkd->nqk", q, k) * scale       # (N, S, S)
    if bipartite:
        same = within_group_mask(S, group_size, q.device)     # (S, S)
        scores = scores.masked_fill(same[None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("nqk,nkd->nqd", attn, v)
    return out, attn


def bipartite_model_fn(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The contract function from the README: (N,S,d)->(N,S,d) NumPy.

    Group size is inferred as S//2 (the goal's fixed layout). Real compute on GPU.
    """
    qt = torch.as_tensor(q, dtype=torch.float32, device=DEVICE)
    kt = torch.as_tensor(k, dtype=torch.float32, device=DEVICE)
    vt = torch.as_tensor(v, dtype=torch.float32, device=DEVICE)
    out, _ = attention(qt, kt, vt, group_size=q.shape[1] // 2, bipartite=True)
    return out.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# Faithful re-implementation of the documented payload contract
# --------------------------------------------------------------------------- #
def run_sweep(batch, bipartite: bool):
    """Mirror task.evaluate's multi-head reshaping, but score from the model's
    real attention weights (and a working softmax)."""
    B, S, d = batch.q.shape
    gs = batch.group_size
    target = torch.as_tensor(batch.target_indices, device=DEVICE)   # (B, S)
    within = within_group_mask(S, gs, DEVICE)                       # (S, S)
    between = ~within

    qn = torch.as_tensor(batch.q, dtype=torch.float32, device=DEVICE)
    kn = torch.as_tensor(batch.k, dtype=torch.float32, device=DEVICE)
    vn = torch.as_tensor(batch.v, dtype=torch.float32, device=DEVICE)

    sweep = []
    for h in NUM_HEADS_SWEEP:
        hd = d // h
        # (B,S,d) -> (B,S,h,hd) -> (B,h,S,hd) -> (B*h,S,hd)  [matches task.py]
        q = qn.reshape(B, S, h, hd).permute(0, 2, 1, 3).reshape(B * h, S, hd)
        k = kn.reshape(B, S, h, hd).permute(0, 2, 1, 3).reshape(B * h, S, hd)
        vv = vn.reshape(B, S, h, hd).permute(0, 2, 1, 3).reshape(B * h, S, hd)

        _, attn = attention(q, k, vv, gs, bipartite)               # (B*h, S, S)
        a4 = attn.reshape(B, h, S, S)

        mean_within = a4[:, :, within].mean().item()
        mean_between = a4[:, :, between].mean().item()
        tgt = target[:, None, :].expand(B, h, S)
        retrieval = (a4.argmax(dim=-1) == tgt).float().mean().item()

        sweep.append({
            "num_heads": h,
            "mean_attn_within": mean_within,
            "mean_attn_between": mean_between,
            "retrieval_acc": retrieval,
        })
    return sweep


# --------------------------------------------------------------------------- #
# Strawman: can a learned CONTENT-only projection do it? (it can OVERFIT but
# cannot GENERALISE — the decisive test).
# --------------------------------------------------------------------------- #
#   The two groups are content-identical (q=k=v=feature_base+noise); the only
#   thing distinguishing the correct cross-group target from the within-group
#   same-feature token is POSITION, which the tokens do not carry. A learned
#   W_q/W_k can memorise one batch's noise (high train acc) but gets ~0 on a
#   fresh seed. The mask, being positional, transfers perfectly.
def _content_scores(X, Wq, Wk, scale):
    return torch.einsum("bqd,bkd->bqk", X @ Wq, X @ Wk) * scale


def train_content_only(train_seeds=(0, 1, 2, 3, 4, 5, 6, 7), test_seed=999,
                       steps=600, lr=1e-2):
    """Train W_q, W_k on several seeds (content only, no mask). Track retrieval
    on the TRAIN seeds vs a HELD-OUT seed each step."""
    torch.manual_seed(0)
    train = []
    for s in train_seeds:
        b = task.generate(seed=s)
        train.append((torch.as_tensor(b.q, dtype=torch.float32, device=DEVICE),
                      torch.as_tensor(b.target_indices, device=DEVICE)))
    bt = task.generate(seed=test_seed)
    Xte = torch.as_tensor(bt.q, dtype=torch.float32, device=DEVICE)
    tte = torch.as_tensor(bt.target_indices, device=DEVICE)

    d = train[0][0].shape[-1]
    eye = torch.eye(d, device=DEVICE)
    Wq = (eye + 0.01 * torch.randn(d, d, device=DEVICE)).clone().requires_grad_(True)
    Wk = (eye + 0.01 * torch.randn(d, d, device=DEVICE)).clone().requires_grad_(True)
    opt = torch.optim.Adam([Wq, Wk], lr=lr)
    scale = 1.0 / (d ** 0.5)

    def acc(X, tgt):
        with torch.no_grad():
            return (_content_scores(X, Wq, Wk, scale).argmax(-1) == tgt).float().mean().item()

    curve = []
    for step in range(steps):
        loss = 0.0
        for X, tgt in train:
            logp = torch.log_softmax(_content_scores(X, Wq, Wk, scale), dim=-1)
            loss = loss - logp.gather(-1, tgt[:, :, None]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            tr = float(np.mean([acc(X, t) for X, t in train]))
            curve.append({"step": step, "train_acc": tr, "heldout_acc": acc(Xte, tte),
                          "loss": float(loss.item())})
    return curve


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
def main():
    run_dir = results_dir(__file__)
    batch = task.generate(seed=42)                # canonical data
    B, S, d = batch.q.shape
    gs = batch.group_size

    # Sanity: the contract function runs and returns the right shape on GPU.
    sample_out = bipartite_model_fn(batch.q, batch.k, batch.v)
    assert sample_out.shape == batch.q.shape, sample_out.shape

    sweep_mech = run_sweep(batch, bipartite=True)
    sweep_base = run_sweep(batch, bipartite=False)
    train_curve = train_content_only()

    # Held-out generalisation: the mask is positional, so it transfers; we report
    # its retrieval on the same fresh seed the content-only model is tested on.
    heldout_batch = task.generate(seed=999)
    mask_heldout = run_sweep(heldout_batch, bipartite=True)
    base_heldout = run_sweep(heldout_batch, bipartite=False)
    mask_heldout_retr = next(r for r in mask_heldout if r["num_heads"] == 4)["retrieval_acc"]
    base_heldout_retr = next(r for r in base_heldout if r["num_heads"] == 4)["retrieval_acc"]

    # Example attention matrices (num_heads=1, batch item 0) for the heatmaps.
    q0 = torch.as_tensor(batch.q, dtype=torch.float32, device=DEVICE)
    k0 = torch.as_tensor(batch.k, dtype=torch.float32, device=DEVICE)
    v0 = torch.as_tensor(batch.v, dtype=torch.float32, device=DEVICE)
    _, attn_mech = attention(q0, k0, v0, gs, bipartite=True)
    _, attn_base = attention(q0, k0, v0, gs, bipartite=False)
    attn_mech0 = attn_mech[0].detach().cpu().numpy()
    attn_base0 = attn_base[0].detach().cpu().numpy()

    # ---- benchmark payload (faithful to the README contract) -------------- #
    payload = {
        "version": 1,
        "config": {
            "group_size": gs,
            "d_model": d,
            "num_features": batch.num_features,
            "batch_size": B,
            "num_heads_sweep": NUM_HEADS_SWEEP,
        },
        "sweep": sweep_mech,
    }
    record_benchmark(__file__, run_dir, payload)

    # ---- artefacts for the Demo tab --------------------------------------- #
    def canon(sw):
        return next(r for r in sw if r["num_heads"] == 4)
    summary = {
        "config": payload["config"],
        "sweep_mechanism": sweep_mech,
        "sweep_baseline": sweep_base,
        "training_content_only": train_curve,
        "canonical": {
            "bipartite_score_mechanism": canon(sweep_mech)["mean_attn_between"]
            - canon(sweep_mech)["mean_attn_within"],
            "bipartite_score_baseline": canon(sweep_base)["mean_attn_between"]
            - canon(sweep_base)["mean_attn_within"],
            "retrieval_mechanism": canon(sweep_mech)["retrieval_acc"],
            "retrieval_baseline": canon(sweep_base)["retrieval_acc"],
            "retrieval_ceiling": 0.5,  # 2 same-feature keys per group -> first-match ceiling
        },
        # The decisive generalisation comparison (all on a held-out seed=999):
        "generalisation": {
            "content_only_train_acc": train_curve[-1]["train_acc"] if train_curve else 0.0,
            "content_only_heldout_acc": train_curve[-1]["heldout_acc"] if train_curve else 0.0,
            "mask_heldout_acc": mask_heldout_retr,
            "baseline_heldout_acc": base_heldout_retr,
        },
        "feature_ids": batch.feature_ids.tolist(),
        "target_indices0": batch.target_indices[0].tolist(),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez(
        run_dir / "attn_example.npz",
        attn_mechanism=attn_base0 * 0 + attn_mech0,  # ensure plain ndarray
        attn_baseline=attn_base0,
        feature_ids=batch.feature_ids,
        group_size=np.int64(gs),
    )

    c, g = summary["canonical"], summary["generalisation"]
    print(f"[bipartite] run_dir = {run_dir}")
    print(f"  bipartite_score  mechanism={c['bipartite_score_mechanism']:.4f}  "
          f"baseline={c['bipartite_score_baseline']:.4f}")
    print(f"  retrieval(canon) mechanism={c['retrieval_mechanism']:.3f}  "
          f"baseline={c['retrieval_baseline']:.3f}  (ceiling 0.5)")
    print(f"  HELD-OUT seed    mask={g['mask_heldout_acc']:.3f}  "
          f"content-only train={g['content_only_train_acc']:.3f} -> heldout={g['content_only_heldout_acc']:.3f}")


if __name__ == "__main__":
    main()
