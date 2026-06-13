"""pass_2 — does a single attention head genuinely *multiply*?

Mechanism (smallest delta from experiments/base_model.py):
  base_model.py's `Attention` forms one query per token via a linear `qkv`
  projection and scores keys by q·k (an *additive* match of features). A head
  built that way can route by `a+b` or by either operand, but cannot single out
  `a·b`. Our head treats the two operands as two query tokens phi(a), phi(b) and
  forms the scoring query by a **bilinear (Hadamard-product) interaction**:

      q = W_o( (W_a phi(a)) ⊙ (W_b phi(b)) )            # ⊙ = elementwise product
      logits_i = beta * < key_i , q >

  The elementwise product is the multiplication: it makes `q` a genuine
  quadratic function of the two operand embeddings, which an additive head
  cannot express. We TRAIN W_a, W_b, W_o, beta with a routing cross-entropy on
  the task's own trial distribution (held-out seeds, never seed 42).

Three model functions are evaluated through the identical task.evaluate harness:
  * trained_fn   — the learned multiplicative head      (recorded to benchmark.json)
  * handbuilt_fn — the SAME bilinear circuit set by hand from the exposed table
                   (a (d,d,d) tensor; hardcoded-weights bonus, mechanism by build)
  * ablation_fn  — the trained head with ⊙ swapped for + (additive). Same weights,
                   multiplication removed -> routing collapses to baseline. This is
                   the causal/faithfulness check.

Everything runs in torch on CUDA.
"""

from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn.functional as F

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback
torch.manual_seed(0)

task = load_task(__file__)

# Fixed, exposed integer embedding table phi: {0..V-1} -> unit vectors in R^d.
TABLE = torch.as_tensor(task.INT_EMBED, dtype=torch.float32, device=DEVICE)  # (V, d)
V, D = TABLE.shape
K_SWEEP = list(task.K_SWEEP)
CANON_K = int(task.CANONICAL_K)
N_POS = int(task.N_POSITIONS)
KMAX = max(K_SWEEP)

# ---------------------------------------------------------------------------
# 1. Build a training set from the task's OWN trial distribution.
#    Different seeds from the eval seed (42) -> honest generalisation test.
# ---------------------------------------------------------------------------
def build_dataset(seeds):
    a_idx, b_idx, cand, tidx = [], [], [], []
    for s in seeds:
        batch = task.generate(seed=int(s))
        a_idx.append(batch.a)
        b_idx.append(batch.b)
        cand.append(batch.candidates)
        tidx.append(batch.true_index)
    a_idx = torch.as_tensor(np.concatenate(a_idx), dtype=torch.long, device=DEVICE)
    b_idx = torch.as_tensor(np.concatenate(b_idx), dtype=torch.long, device=DEVICE)
    cand = torch.as_tensor(np.concatenate(cand), dtype=torch.long, device=DEVICE)
    tidx = torch.as_tensor(np.concatenate(tidx), dtype=torch.long, device=DEVICE)
    return a_idx, b_idx, cand, tidx


train_seeds = list(range(1, 31))        # 30 batches, ~30k trials, all K
val_seeds = list(range(101, 106))       # disjoint validation
tr_a, tr_b, tr_cand, tr_t = build_dataset(train_seeds)
va_a, va_b, va_cand, va_t = build_dataset(val_seeds)

# ---------------------------------------------------------------------------
# 2. The trainable bilinear-query head.
# ---------------------------------------------------------------------------
R = 1024  # rank of the multiplicative interaction
Wa = torch.empty(R, D, device=DEVICE).normal_(0, D ** -0.5).requires_grad_(True)
Wb = torch.empty(R, D, device=DEVICE).normal_(0, D ** -0.5).requires_grad_(True)
Wo = torch.empty(D, R, device=DEVICE).normal_(0, R ** -0.5).requires_grad_(True)
log_beta = torch.tensor(float(np.log(10.0)), device=DEVICE, requires_grad=True)


def head_query(a_emb, b_emb, multiplicative=True):
    """Form the scoring query from two operand embeddings.

    a_emb,b_emb: (..., D). multiplicative=True -> Hadamard product (the real
    mechanism); False -> additive (the ablation), same weights.
    """
    qa = a_emb @ Wa.t()          # (..., R)
    qb = b_emb @ Wb.t()          # (..., R)
    inter = qa * qb if multiplicative else qa + qb
    return inter @ Wo.t()        # (..., D)


def logits_from(a_emb, b_emb, keys, beta, multiplicative=True):
    q = head_query(a_emb, b_emb, multiplicative)          # (B, D)
    return beta * torch.einsum("bpd,bd->bp", keys, q)     # (B, n_pos)


def accuracy(a_idx, b_idx, cand, tidx, multiplicative=True):
    with torch.no_grad():
        a_emb = TABLE[a_idx]
        b_emb = TABLE[b_idx]
        keys = TABLE[cand]
        beta = log_beta.exp()
        lg = logits_from(a_emb, b_emb, keys, beta, multiplicative)
        return (lg.argmax(1) == tidx).float().mean().item()


opt = torch.optim.Adam([Wa, Wb, Wo, log_beta], lr=3e-3)
N = tr_a.shape[0]
BS = 2048
STEPS = 2500
curve_steps, curve_val = [], []

for step in range(STEPS):
    sel = torch.randint(0, N, (BS,), device=DEVICE)
    a_emb = TABLE[tr_a[sel]]
    b_emb = TABLE[tr_b[sel]]
    keys = TABLE[tr_cand[sel]]
    beta = log_beta.exp().clamp(max=60.0)
    lg = logits_from(a_emb, b_emb, keys, beta, multiplicative=True)
    loss = F.cross_entropy(lg, tr_t[sel])
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step % 250 == 0 or step == STEPS - 1:
        v = accuracy(va_a, va_b, va_cand, va_t, multiplicative=True)
        curve_steps.append(step)
        curve_val.append(v)
        print(f"step {step:5d}  loss {loss.item():.4f}  val_acc {v:.3f}  beta {beta.item():.2f}")

BETA = float(log_beta.exp().clamp(max=60.0).item())
Wa_f = Wa.detach()
Wb_f = Wb.detach()
Wo_f = Wo.detach()

# ---------------------------------------------------------------------------
# 3. Hand-built bilinear circuit (mechanism by construction; bonus).
#    A genuine (d,d,d) bilinear tensor T contracting phi(a)⊗phi(b) -> phi(a*b).
#    Using dual (pseudo-inverse) operand bases makes the in-range selection exact,
#    so this is a true multiplicative circuit, NOT an argmax-decode + table lookup.
# ---------------------------------------------------------------------------
def build_handbuilt_tensor():
    ops = torch.arange(0, KMAX, device=DEVICE)              # operands 0..KMAX-1
    A = TABLE[ops]                                          # (KMAX, D)
    # dual basis: Atil @ A.T = I  =>  Atil = (A A^T)^{-1} A
    Atil = torch.linalg.solve(A @ A.t(), A)                # (KMAX, D)
    prod_ids = (ops[:, None] * ops[None, :]).reshape(-1)   # (KMAX*KMAX,)
    P = TABLE[prod_ids].reshape(KMAX, KMAX, D)             # phi(a*b), (KMAX,KMAX,D)
    # T[k,i,j] = sum_{a,b} phi(a*b)_k * Atil[a]_i * Atil[b]_j
    T = torch.einsum("abk,ai,bj->kij", P, Atil, Atil)      # (D, D, D)
    return T


T_HB = build_handbuilt_tensor()
BETA_HB = 30.0

# ---------------------------------------------------------------------------
# 4. The three model_fn variants (signature: a_vec, b_vec, key_vecs -> logits).
# ---------------------------------------------------------------------------
def trained_fn(a_vec, b_vec, key_vecs):
    a = torch.as_tensor(a_vec, dtype=torch.float32, device=DEVICE)
    b = torch.as_tensor(b_vec, dtype=torch.float32, device=DEVICE)
    keys = torch.as_tensor(key_vecs, dtype=torch.float32, device=DEVICE)
    inter = (a @ Wa_f.t()) * (b @ Wb_f.t())
    q = inter @ Wo_f.t()
    return (BETA * (keys @ q)).detach().cpu().numpy()


def ablation_fn(a_vec, b_vec, key_vecs):
    # Same trained weights, Hadamard product replaced by SUM (additive).
    a = torch.as_tensor(a_vec, dtype=torch.float32, device=DEVICE)
    b = torch.as_tensor(b_vec, dtype=torch.float32, device=DEVICE)
    keys = torch.as_tensor(key_vecs, dtype=torch.float32, device=DEVICE)
    inter = (a @ Wa_f.t()) + (b @ Wb_f.t())
    q = inter @ Wo_f.t()
    return (BETA * (keys @ q)).detach().cpu().numpy()


def handbuilt_fn(a_vec, b_vec, key_vecs):
    a = torch.as_tensor(a_vec, dtype=torch.float32, device=DEVICE)
    b = torch.as_tensor(b_vec, dtype=torch.float32, device=DEVICE)
    keys = torch.as_tensor(key_vecs, dtype=torch.float32, device=DEVICE)
    pred = torch.einsum("kij,i,j->k", T_HB, a, b)          # bilinear contraction
    return (BETA_HB * (keys @ pred)).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# 5. Evaluate. The trained head is the headline -> recorded to benchmark.json.
# ---------------------------------------------------------------------------
payload = task.evaluate(trained_fn)
run_dir = results_dir(__file__)
record_benchmark(__file__, run_dir, payload)

# Companion evaluations (saved as extras for the Demo viz, not the benchmark).
payload_ab = task.evaluate(ablation_fn)
payload_hb = task.evaluate(handbuilt_fn)


def per_k(p):
    return {int(r["k"]): r for r in p["sweep"]}


sw_tr, sw_ab, sw_hb = per_k(payload), per_k(payload_ab), per_k(payload_hb)
base = {int(r["k"]): r for r in payload["linear_baseline"]}

extras = {
    "k_sweep": K_SWEEP,
    "canonical_k": CANON_K,
    "n_positions": N_POS,
    "beta_trained": BETA,
    "rank": R,
    "series": {
        "trained":   [sw_tr[k]["routing_accuracy"] for k in K_SWEEP],
        "handbuilt": [sw_hb[k]["routing_accuracy"] for k in K_SWEEP],
        "ablation":  [sw_ab[k]["routing_accuracy"] for k in K_SWEEP],
        "baseline":  [base[k]["routing_accuracy"] for k in K_SWEEP],
    },
    "attended_mass_trained": [sw_tr[k]["attended_mass"] for k in K_SWEEP],
    "train_curve": {"steps": curve_steps, "val_acc": curve_val},
}

# Canonical-K scatter: predicted attended integer value vs true product.
batch = task.generate(seed=task.EVAL_SEED)
mask = batch.k == CANON_K
true_prod, pred_val, correct = [], [], []
for i in np.where(mask)[0]:
    a_vec = task.embed(batch.a[i])
    b_vec = task.embed(batch.b[i])
    key_vecs = task.embed(batch.candidates[i])
    lg = trained_fn(a_vec, b_vec, key_vecs)
    pj = int(np.argmax(lg))
    true_prod.append(int(batch.product[i]))
    pred_val.append(int(batch.candidates[i][pj]))
    correct.append(bool(pj == int(batch.true_index[i])))
extras["scatter"] = {
    "true_product": true_prod, "pred_value": pred_val, "correct": correct,
}

with open(run_dir / "extras.json", "w") as f:
    json.dump(extras, f, indent=2)

print("\n=== summary ===")
print(f"mean routing acc (trained)   : {np.mean(extras['series']['trained']):.3f}")
print(f"mean routing acc (handbuilt) : {np.mean(extras['series']['handbuilt']):.3f}")
print(f"mean routing acc (ablation + ): {np.mean(extras['series']['ablation']):.3f}")
print(f"mean routing acc (baseline)  : {np.mean(extras['series']['baseline']):.3f}")
print(f"canonical attended_mass      : {sw_tr[CANON_K]['attended_mass']:.3f}")
print(f"results in {run_dir}")
