"""
attention_sat / pass_3 — Saturation as softmax-Jacobian (gradient) collapse,
with a no-exp ablation strawman.

Mechanistic claim
-----------------
Attention saturation is *defined* by the softmax pushing probability mass onto a
few positions as logits grow, which collapses the softmax Jacobian and makes
gradients vanish. The per-query softmax Jacobian is  J = diag(p) - p pᵀ, whose
trace is  tr(J) = 1 - Σ_i p_i².  So the attention **concentration**

        C = Σ_i p_i²   = 1 - tr(J)

is, mechanistically, "one minus the gradient flow". C → 1/seq_len (uniform,
gradients alive) in the linear regime and C → 1 (one-hot, gradients dead) under
saturation. We use C as `saturation_score`. We separately *measure* the real
gradient with GPU autograd (∂‖attn·v‖²/∂logits) to confirm the collapse — this
is the faithfulness / causal evidence: the same quantity that detects the regime
is the quantity that kills the gradient.

Failing strawman (ablation of `exp`)
------------------------------------
Replace softmax with relu-linear normalization  w_i = relu(s_i)/Σ relu(s_j).
Because it has no `exp`, multiplying logits by `logit_scale` factors out of the
ratio: the weights are **scale-invariant**. The no-exp head therefore *cannot*
saturate — its concentration is flat across the whole sweep and its AUROC
collapses to chance. This isolates the `exp` nonlinearity as the mechanism.

Everything runs in torch on CUDA (forward + autograd).
"""
import json
import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a visible GPU; no CPU fallback
task = load_task(__file__)


# --------------------------------------------------------------------------- #
# GPU attention primitives                                                    #
# --------------------------------------------------------------------------- #
def _to_t(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype, device=DEVICE)


def _weights(q, k, logit_scale, causal_mask, use_exp=True):
    """Return attention weights (batch, seq, seq) on GPU.

    use_exp=True  -> standard softmax (can saturate).
    use_exp=False -> relu-linear normalization (no exp; scale-invariant).
    """
    qt, kt = _to_t(q), _to_t(k)
    raw = torch.einsum("bqd,bkd->bqk", qt, kt) * float(logit_scale)
    m = _to_t(causal_mask, dtype=torch.bool) if causal_mask is not None else None

    if use_exp:
        logits = raw if m is None else torch.where(m, raw, torch.full_like(raw, -1e9))
        return torch.softmax(logits, dim=-1)

    s = torch.relu(raw)
    if m is not None:
        s = torch.where(m, s, torch.zeros_like(s))
    return s / (s.sum(dim=-1, keepdim=True) + 1e-12)


def _entropy(w):
    return -(w * torch.log(w + 1e-12)).sum(dim=-1)


def _concentration(w):
    """Σ p²  per (batch, query) == 1 - tr(softmax Jacobian)."""
    return (w * w).sum(dim=-1)


def _grad_collapse(q, k, v, logit_scale, causal_mask):
    """Real GPU autograd: mean |∂ loss / ∂ logits| over valid positions.

    loss = ½‖attn·v‖².  As attention saturates the softmax Jacobian collapses
    and this gradient vanishes — the mechanistic signature of saturation.
    """
    qt, kt, vt = _to_t(q), _to_t(k), _to_t(v)
    raw = (torch.einsum("bqd,bkd->bqk", qt, kt) * float(logit_scale)).detach()
    raw.requires_grad_(True)
    m = _to_t(causal_mask, dtype=torch.bool) if causal_mask is not None else None
    logits = raw if m is None else torch.where(m, raw, torch.full_like(raw, -1e9))
    w = torch.softmax(logits, dim=-1)
    o = torch.einsum("bqk,bkd->bqd", w, vt)
    loss = 0.5 * (o * o).sum()
    loss.backward()
    g = raw.grad.abs()
    if m is not None:
        g = g[m.unsqueeze(0).expand_as(g)]
    return float(g.mean().item())


# --------------------------------------------------------------------------- #
# The attempt's model_fn (contract with task.evaluate)                        #
# --------------------------------------------------------------------------- #
def real_model_fn(q, k, v, logit_scale, causal_mask):
    """Exact softmax attention; saturation_score = mean attention concentration
    Σp²  (== 1 - mean softmax-Jacobian trace)."""
    w = _weights(q, k, logit_scale, causal_mask, use_exp=True)
    ent = _entropy(w)
    conc = _concentration(w)
    return {
        "attn_weights": w.detach().cpu().numpy().astype(np.float32),
        "attn_entropy": ent.detach().cpu().numpy().astype(np.float32),
        "saturation_score": float(conc.mean().item()),
    }


# --------------------------------------------------------------------------- #
# Visualisation / contrast artefacts                                          #
# --------------------------------------------------------------------------- #
def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    c = 0.0
    for p in pos:
        c += float(np.sum(neg < p)) + 0.5 * float(np.sum(neg == p))
    return float(c / (len(pos) * len(neg)))


def build_viz(run_dir):
    batch = task.generate(seed=0)
    q, k, v = batch.q, batch.k, batch.v
    mask = batch.causal_mask
    scales = [float(s) for s in batch.logit_scales]
    labels = [1 if s >= 10.0 else 0 for s in scales]

    real_conc, real_ent, real_grad, real_maxw, mean_attn = [], [], [], [], []
    straw_conc, straw_ent = [], []

    for s in scales:
        wr = _weights(q, k, s, mask, use_exp=True)
        real_conc.append(float(_concentration(wr).mean().item()))
        real_ent.append(float(_entropy(wr).mean().item()))
        real_maxw.append(float(wr.max(dim=-1).values.mean().item()))
        real_grad.append(_grad_collapse(q, k, v, s, mask))
        mean_attn.append(wr.mean(dim=0).detach().cpu().numpy().astype(np.float32).tolist())

        ws = _weights(q, k, s, mask, use_exp=False)
        straw_conc.append(float(_concentration(ws).mean().item()))
        straw_ent.append(float(_entropy(ws).mean().item()))

    viz = {
        "scales": scales,
        "labels": labels,
        "threshold": 10.0,
        "real": {
            "concentration": real_conc,
            "mean_entropy": real_ent,
            "grad_norm": real_grad,
            "max_weight": real_maxw,
            "auroc": _auroc(real_conc, labels),
        },
        "strawman": {
            "concentration": straw_conc,
            "mean_entropy": straw_ent,
            "auroc": _auroc(straw_conc, labels),
            "name": "no-exp (relu-linear) ablation",
        },
        "mean_attn": mean_attn,  # 7 x (seq x seq)
    }
    (run_dir / "viz.json").write_text(json.dumps(viz))
    return viz


def main():
    payload = task.evaluate(real_model_fn)
    run_dir = results_dir(__file__)
    record_benchmark(__file__, run_dir, payload)
    viz = build_viz(run_dir)
    print(
        f"[pass_3] real AUROC={viz['real']['auroc']:.3f}  "
        f"strawman(no-exp) AUROC={viz['strawman']['auroc']:.3f}  "
        f"grad {viz['real']['grad_norm'][0]:.2e} -> {viz['real']['grad_norm'][-1]:.2e}"
    )


if __name__ == "__main__":
    main()
