"""pass_6 — hand-set single attention head for modular addition (a+b) mod p.

Type: hand_built (a real base_model-style head with hand-set weights, NO training).

Architecture = base_model.py reduced to its smallest relevant delta: a token
embedding E[vocab, d_model] and two linear projections W_Q, W_K. One head, no MLP,
no softmax is needed because the task only reads the per-position Q/K vectors.
vocab = p+1 (the '=' separator id == p).

THE ADDITION HEAD (submitted to benchmark.json)
  Per frequency k = 1..p//2 a dedicated channel pair (even=cos, odd=sin):
    E[x] : even = cos(2 pi k x / p), odd = sin(2 pi k x / p)
    Q(a) = E[a]                      (W_Q = identity)
    K(b) = E[b] * s,  s negates ONLY the sin/odd channels
  => q(a).k(b) = sum_k [ cos(ka)cos(kb) - sin(ka)sin(kb) ] = sum_k cos(k(a+b)),
  which depends ONLY on (a+b) mod p and peaks when a+b is constant. This is the
  genuine modular-ADDITION circuit.

WHY phase_error sits at ~pi/2 for this head (and that is intrinsic, not a bug).
  Writing each channel as a complex direction u_c (Re=cos-weight, Im=sin-weight),
  q.k splits into an (a+b) term proportional to sum_c u_c*v_c and an (a-b) term
  proportional to sum_c u_c*conj(v_c). The metric's phase_error is 0 iff
  v_c is a positive multiple of u_c for every channel, which forces
  sum_c u_c*conj(v_c) > 0 — i.e. a DIFFERENCE (a-b) head. A pure (a+b) head must
  therefore carry per-channel phase pi on its sin channels, and the magnitude-
  weighted mean lands at exactly pi/2. So for THIS metric:
      - headline `fourier_alignment` is maxed (==1) by the addition head;
      - `phase_error`==0 is achievable ONLY by the (a-b) difference head (K=Q).
  We submit the addition head and SHOW the difference head reaching phase 0 as a
  labelled contrast, so the trade-off is explicit rather than hidden.

CONTRASTS saved as artifacts (not scored): the (a-b) difference head (phase 0) and
a random non-Fourier strawman (alignment collapses to the 2/d_head baseline).

OPERATING RANGE: the same hand construction is swept over primes p that fit in
d_head=128 (p<=113), reporting alignment/phase at each, to show the mechanism is
not p=97-specific.

All real compute runs on CUDA.
"""

import json
import numpy as np
import torch
from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # pipeline guarantees a GPU; no CPU fallback
P = 97
D_HEAD = 128


def _embed_table(p: int) -> torch.Tensor:
    """E[vocab=p+1, D_HEAD] on GPU. even=cos, odd=sin per frequency; '=' row = 0."""
    n_freq = p // 2
    ks = torch.arange(1, n_freq + 1, device=DEVICE, dtype=torch.float32)
    x = torch.arange(p, device=DEVICE, dtype=torch.float32)
    ang = 2.0 * np.pi * torch.outer(x, ks) / p          # [p, n_freq]
    E = torch.zeros(p + 1, D_HEAD, device=DEVICE)
    E[:p, 0:2 * n_freq:2] = torch.cos(ang)
    E[:p, 1:2 * n_freq:2] = torch.sin(ang)
    return E


def _key_sign(p: int, kind: str) -> torch.Tensor:
    """Diagonal of W_K. 'add' negates sin channels (-> a+b); 'diff' is identity (-> a-b)."""
    n_freq = p // 2
    s = torch.ones(D_HEAD, device=DEVICE)
    s[2 * n_freq:] = 0.0
    if kind == "add":
        s[1:2 * n_freq:2] = -1.0          # negate sin/odd channels
    return s


def _head_fn(p: int, kind: str):
    """Return a contract model_fn (NumPy in/out) for a hand-set head on modulus p.

    'add'  : Fourier head, K negates sin channels  -> genuine (a+b), alignment 1.
    'diff' : Fourier head with K==Q                 -> (a-b), phase 0 (degenerate).
    'rand' : INDEPENDENT random Q/K embeddings      -> no Fourier structure, fails.
    """
    if kind == "rand":
        gq = torch.Generator(device="cpu").manual_seed(0)
        gk = torch.Generator(device="cpu").manual_seed(1)
        Eq = torch.randn(p + 1, D_HEAD, generator=gq).to(DEVICE)
        Ek = torch.randn(p + 1, D_HEAD, generator=gk).to(DEVICE)
    else:
        Eq = _embed_table(p)
        Ek = Eq * _key_sign(p, kind)       # W_K = diag(s) folded into the table

    def fn(tokens: np.ndarray):
        tok = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)
        Q = Eq[tok]                        # W_Q = identity
        K = Ek[tok]
        return Q.detach().cpu().numpy(), K.detach().cpu().numpy()
    return fn


def _grid(p: int):
    a = np.repeat(np.arange(p), p).astype(np.int64)
    b = np.tile(np.arange(p), p).astype(np.int64)
    eq = np.full(p * p, p, dtype=np.int64)
    return np.stack([a, b, eq], axis=1), a, b


def _sweep_for(task, p: int, kind: str):
    """Build the head on GPU, run task._compute_sweep (CPU) for modulus p."""
    tokens, a, b = _grid(p)
    Q_all, K_all = _head_fn(p, kind)(tokens)
    sweep = task._compute_sweep(Q_all[:, 0, :], K_all[:, 1, :], a, b, p)
    al = [r["alignment"] for r in sweep]
    ph = [r["phase_error"] for r in sweep]
    return sweep, al, ph


def main():
    task = load_task(__file__)
    run_dir = results_dir(__file__)

    # ---- Scored payload: the genuine ADDITION head ----------------------------
    payload = task.evaluate(_head_fn(P, "add"))
    record_benchmark(__file__, run_dir, payload)
    add_sweep = payload["sweep"]

    # ---- Contrasts (artifacts only) ------------------------------------------
    _, diff_al, diff_ph = _sweep_for(task, P, "diff")     # phase -> 0
    _, rand_al, rand_ph = _sweep_for(task, P, "rand")     # alignment -> baseline

    # ---- Operating range over primes that fit in d_head=128 ------------------
    op = []
    for q in [11, 23, 47, 73, 97, 113]:
        _, al, ph = _sweep_for(task, q, "add")
        op.append({
            "modulus": int(q),
            "mean_alignment": float(np.mean(al)),
            "max_alignment": float(np.max(al)),
            "mean_phase": float(np.mean(ph)),
        })

    artifact = {
        "modulus": P,
        "d_head": D_HEAD,
        "random_baseline_alignment": 2.0 / D_HEAD,
        "freq": [r["frequency"] for r in add_sweep],
        "add": {
            "alignment": [r["alignment"] for r in add_sweep],
            "phase_error": [r["phase_error"] for r in add_sweep],
            "explained_variance": [r["explained_variance"] for r in add_sweep],
        },
        "diff": {"alignment": diff_al, "phase_error": diff_ph},
        "rand": {"alignment": rand_al, "phase_error": rand_ph},
        "operating_range": op,
        "max_alignment": payload["max_alignment"],
        "argmax_alignment_freq": payload["argmax_alignment_freq"],
        "total_explained_variance": payload["total_explained_variance"],
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifact))

    print(f"ADD  mean_align={np.mean(artifact['add']['alignment']):.4f} "
          f"phase={np.mean(artifact['add']['phase_error']):.4f} "
          f"max_align={payload['max_alignment']:.4f}")
    print(f"DIFF mean_align={np.mean(diff_al):.4f} phase={np.mean(diff_ph):.4f}")
    print(f"RAND mean_align={np.mean(rand_al):.4f} (baseline {2.0/D_HEAD:.4f})")
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
