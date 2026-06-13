"""
attention_histogram / pass_3 — TRAINED gated-denoising attention head.

Why this is different from pass_2 (and from the plain dot-product baseline)
-------------------------------------------------------------------------
pass_2 *hand-set* an iterative power-iteration. Its faithfulness story was
trivial ("the circuit uses the mechanism because I wrote the mechanism"), and
the power-iteration framing drifted from a transformer. Here we keep a minimal
**two-attention-block** circuit that mirrors `base_model.py` and let **gradient
descent discover** how much to denoise — including a learned **gate** that turns
the denoising OFF when it would only inject noise.

The circuit (delta from base_model.py)
--------------------------------------
The target direction is uniformly random every example, so the data is
rotation-equivariant and every learnable Q/K weight matrix collapses to a
scalar. We therefore expose the handful of rotation-invariant scalars a gated
2-block attention can use:

  block 1  (denoise, a GATED residual write to the stream):
      a     = softmax(beta1 . Kn q)        # soft attention with the noisy query
      cen   = Kn^T a                        # key centroid (the denoised target est.)
      r     = ||cen||  in [0,1]             # resultant length = key concentration
      g     = sigmoid(gate_w . (r - gate_b))# gate: ~0 if keys diffuse, ~1 if clustered
      q'    = q + gamma . g . cen           # gated residual update
  block 2  (score / readout):
      logits = beta2 . (Kn q')             # final attention logits

Why this is the right mechanism: with cosine `sim` to the target, distractors
sit at cosine `~sim^2` to *each other* but `sim` to the target, so the **target
is the most central key** and `cen` points at it — but only when the keys
actually cluster. The gate reads that off the resultant length `r`: at `sim=0`
the keys are near-orthogonal, `r ~ 0.25`, the gate closes, and the readout falls
back to the (best-available) raw query so the head does NOT regress below
dot-product; at high `sim` the keys cluster, `r` is large, the gate opens, and
denoising rescues an aim that dot-product loses entirely.

`beta1, beta2, gamma, gate_w, gate_b` are TRAINED by cross-entropy on fresh
batches (seeds disjoint from EVAL_SEED=7), then frozen. Trained attempt.

Causal / faithfulness controls evaluated in this file:
  * mechanism   : trained scalars, n_iter=1
  * ablation    : n_iter=0  -> temperature-only readout = beta2.(Kn q).
                  Its argmax equals dot-product's, so it is *sharp but its
                  targeting collapses to baseline* — isolating that the gated
                  denoising block (not the temperature) is what fixes aim.
  * baseline    : plain softmax(K q), supplied by task.evaluate.

All compute runs in torch on CUDA (training + inference).
"""

import argparse
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # framework guarantees a visible GPU; no CPU fallback.

task = load_task(__file__)


# --------------------------------------------------------------------------- #
# Model: rotation-equivariant gated-denoising 2-block attention.
# --------------------------------------------------------------------------- #
def make_model_fn(p: dict, n_iter: int):
    """Return model_fn: (query (d,), keys (n,d)) -> logits (n,), on GPU.

    n_iter=0 removes the denoising block -> temperature-only dot product.
    """
    b1, b2 = p["beta1"], p["beta2"]
    gamma, gw, gb = p["gamma"], p["gate_w"], p["gate_b"]

    def model_fn(query: np.ndarray, keys: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(query, dtype=torch.float32, device=DEVICE)
        k = torch.as_tensor(keys, dtype=torch.float32, device=DEVICE)
        qc = q / (q.norm() + 1e-8)
        kn = k / (k.norm(dim=1, keepdim=True) + 1e-8)        # (n,d) unit keys
        for _ in range(n_iter):
            a = torch.softmax(b1 * (kn @ qc), dim=0)         # (n,) attention
            cen = kn.t() @ a                                 # (d,) key centroid
            r = cen.norm()                                   # concentration in [0,1]
            g = torch.sigmoid(gw * (r - gb))                 # gate
            qc = qc + gamma * g * cen                        # gated residual update
        logits = b2 * (kn @ qc)                              # (n,) logits
        return logits.detach().cpu().numpy()

    return model_fn


# --------------------------------------------------------------------------- #
# Training: discover the scalars by cross-entropy on fresh batches.
# --------------------------------------------------------------------------- #
def train(n_iter: int, steps: int, lr: float, n_train_seeds: int):
    torch.manual_seed(0)

    # Gather training examples from seeds disjoint from EVAL_SEED (=7).
    Q, K, T = [], [], []
    for sd in range(1000, 1000 + n_train_seeds):
        b = task.generate(seed=sd)
        for q, k, t in zip(b.queries, b.keys, b.target_index):
            Q.append(q)
            K.append(k)
            T.append(t)
    Qt = torch.as_tensor(np.stack(Q), dtype=torch.float32, device=DEVICE)   # (N,d)
    Kt = torch.as_tensor(np.stack(K), dtype=torch.float32, device=DEVICE)   # (N,n,d)
    Tt = torch.as_tensor(np.asarray(T), dtype=torch.long, device=DEVICE)    # (N,)

    Qn = Qt / (Qt.norm(dim=1, keepdim=True) + 1e-8)
    Kn = Kt / (Kt.norm(dim=2, keepdim=True) + 1e-8)

    sp = torch.nn.functional.softplus
    # Positive params via softplus; gate bias is free. Sensible inits.
    raw_b1 = torch.tensor(1.0, device=DEVICE, requires_grad=True)   # ~beta1 1.3
    raw_b2 = torch.tensor(2.0, device=DEVICE, requires_grad=True)   # ~beta2 2.1
    raw_g = torch.tensor(1.0, device=DEVICE, requires_grad=True)    # ~gamma 1.3
    raw_gw = torch.tensor(2.0, device=DEVICE, requires_grad=True)   # ~gate_w 2.1
    gate_b = torch.tensor(0.45, device=DEVICE, requires_grad=True)  # gate bias
    opt = torch.optim.Adam([raw_b1, raw_b2, raw_g, raw_gw, gate_b], lr=lr)

    history = []
    for step in range(steps):
        b1, b2, gamma, gw = sp(raw_b1), sp(raw_b2), sp(raw_g), sp(raw_gw)
        qc = Qn
        for _ in range(n_iter):
            scores = torch.einsum("nkd,nd->nk", Kn, qc)      # (N,n)
            a = torch.softmax(b1 * scores, dim=1)
            cen = torch.einsum("nk,nkd->nd", a, Kn)          # (N,d)
            r = cen.norm(dim=1, keepdim=True)                # (N,1)
            g = torch.sigmoid(gw * (r - gate_b))             # (N,1)
            qc = qc + gamma * g * cen
        logits = b2 * torch.einsum("nkd,nd->nk", Kn, qc)     # (N,n)
        loss = torch.nn.functional.cross_entropy(logits, Tt)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 20) == 0 or step == steps - 1:
            history.append(float(loss.detach().cpu()))

    p = {
        "beta1": float(sp(raw_b1).detach().cpu()),
        "beta2": float(sp(raw_b2).detach().cpu()),
        "gamma": float(sp(raw_g).detach().cpu()),
        "gate_w": float(sp(raw_gw).detach().cpu()),
        "gate_b": float(gate_b.detach().cpu()),
    }
    return p, history


def softmax_np(x):
    z = np.asarray(x, np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_iter", type=int, default=1,
                    help="denoising blocks in the trained mechanism")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n_train_seeds", type=int, default=80,
                    help="training seeds (each yields 80 examples)")
    args = ap.parse_args()

    # --- Train the mechanism's scalars. ---
    p, hist = train(args.n_iter, args.steps, args.lr, args.n_train_seeds)

    mech_fn = make_model_fn(p, args.n_iter)
    ablate_fn = make_model_fn(p, 0)   # denoising block removed (temperature only)

    payload = task.evaluate(mech_fn)
    payload["model_name"] = (
        f"trained_gated_denoise(beta1={p['beta1']:.2f},beta2={p['beta2']:.2f},"
        f"gamma={p['gamma']:.2f},gate_w={p['gate_w']:.2f},"
        f"gate_b={p['gate_b']:.2f},n_iter={args.n_iter})")
    ablate_payload = task.evaluate(ablate_fn)

    run_dir = results_dir(__file__)

    # --- Depth / operating-range sweep: hit & sharpness vs number of blocks. ---
    depth = []
    for ni in [0, 1, 2, 3]:
        pv = task.evaluate(make_model_fn(p, ni))
        canon = next(s for s in pv["sweep"]
                     if abs(s["similarity"] - pv["canonical_similarity"]) < 1e-9)
        depth.append({
            "n_iter": ni,
            "mean_hit": float(np.mean([s["target_hit_rate"] for s in pv["sweep"]])),
            "canonical_sharpness": float(canon["attention_sharpness"]),
            "canonical_hit": float(canon["target_hit_rate"]),
        })

    # --- Per-slice example histograms for the Demo tab. ---
    batch = task.generate(seed=task.EVAL_SEED)
    examples = []
    for ci, sim in enumerate(task.KEY_SIM_SWEEP):
        idx = ci * task.N_SEEDS
        q, k, tgt = batch.queries[idx], batch.keys[idx], batch.target_index[idx]
        examples.append({
            "similarity": float(sim),
            "target_index": int(tgt),
            "mech_attn": softmax_np(mech_fn(q, k)).tolist(),
            "ablate_attn": softmax_np(ablate_fn(q, k)).tolist(),
            "base_attn": softmax_np(
                k.astype(np.float64) @ q.astype(np.float64)).tolist(),
        })

    params = dict(p)
    params["n_iter"] = args.n_iter
    params["loss_history"] = hist

    with open(run_dir / "payload.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(run_dir / "ablation.json", "w") as f:
        json.dump(ablate_payload, f, indent=2)
    with open(run_dir / "examples.json", "w") as f:
        json.dump({"n_positions": task.N_POSITIONS, "examples": examples,
                   "depth": depth, "params": params}, f, indent=2)

    record_benchmark(__file__, run_dir, payload)

    print(f"model={payload['model_name']}")
    print(f"  learned {p}")
    for s, ab, b in zip(payload["sweep"], ablate_payload["sweep"],
                        payload["linear_baseline"]):
        print(f"  sim={s['similarity']:.1f} | hit mech={s['target_hit_rate']:.2f}"
              f" ablate={ab['target_hit_rate']:.2f} base={b['target_hit_rate']:.2f}"
              f" | sharp mech={s['attention_sharpness']:.2f}"
              f" base={b['attention_sharpness']:.2f}")


if __name__ == "__main__":
    main()
