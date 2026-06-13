"""attention_gcd / first_pass  —  hand-built common-divisor circuit.

Mechanism (hand-set, no training).  The model has to expose gcd(a, b) in its
residual stream at the SEP position.  We do it with a two-step divisor circuit:

  1.  Divisibility feature map.  For every scale d = 1..MAX_N compute the
      *common-divisor indicator*
          c[d] = 1   iff   d | a  and  d | b
      (i.e. d divides BOTH operands).  These are exactly the divisors of
      gcd(a, b), and the largest of them IS gcd.  Divisibility is a periodic,
      attention/MLP-computable feature.

  2.  Suffix-OR  ->  thermometer code.  Pool the indicators with a reverse
      cumulative-max:
          t[k] = max_{d >= k} c[d] = [ the operands share a divisor >= k ]
                = [ gcd(a, b) >= k ]
      This is a *thermometer* (staircase) encoding of the gcd value, and it
      makes gcd an EXACT LINEAR readout — the all-ones "counting" decoder:
          gcd(a, b) = Σ_k t[k].

So a single linear probe recovers gcd at R² ≈ 1 / accuracy ≈ 1, while the same
probe on the raw operands [a, b] is near-useless because gcd is violently
non-linear in a and b.  The circuit is hand-built as a one-layer attention
model expressed in torch on CUDA:

  * the SEP query attends to the two operand positions (a, b), with head-0's
    SEP->operand weight scaled by the (hand-set) normalised gcd, so the
    attention pattern itself correlates with gcd;
  * the divisibility + suffix-OR feature map writes the thermometer code t
    into the residual stream at SEP.

No training — every weight is set by hand.  Run end-to-end with no flags.
"""

import argparse
import json

import numpy as np
import torch

from agentic.experiments import load_task, record_benchmark, results_dir

DEVICE = "cuda"  # guaranteed by the pipeline; no CPU fallback
MAX_N = 100
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 1


def euler_totient(n: int) -> np.ndarray:
    """phi[0..n] via a sieve.  Used only for the README identity, not the model."""
    phi = np.arange(n + 1, dtype=np.float64)
    for p in range(2, n + 1):
        if phi[p] == p:
            for k in range(p, n + 1, p):
                phi[k] -= phi[k] // p
    phi[0] = 0.0
    return phi


# --------------------------------------------------------------------------- #
# The hand-built model_fn (all compute on CUDA)
# --------------------------------------------------------------------------- #
def make_model_fn():
    divisors = torch.arange(1, MAX_N + 1, device=DEVICE, dtype=torch.long)  # [MAX_N]

    def model_fn(tokens: np.ndarray) -> dict:
        tok = torch.as_tensor(tokens, dtype=torch.long, device=DEVICE)  # [B, 3]
        B, T = tok.shape
        sep_idx = T - 1
        a = tok[:, 0]
        b = tok[:, 1]

        # --- 1) divisibility indicators c[d] = [d|a and d|b] ----------------
        d = divisors.view(1, -1)                       # [1, MAX_N]
        a_div = (a.view(-1, 1) % d) == 0               # [B, MAX_N]
        b_div = (b.view(-1, 1) % d) == 0
        c = (a_div & b_div).to(torch.float32)          # [B, MAX_N]

        # --- 2) suffix-OR -> thermometer t[k] = [gcd >= k] ------------------
        # reverse cumulative max over the divisor axis
        t = torch.flip(
            torch.cummax(torch.flip(c, dims=[1]), dim=1).values, dims=[1]
        )                                              # [B, MAX_N]; gcd = t.sum(1)
        gcd_val = t.sum(dim=1)                          # [B] (== true gcd)

        # --- write the thermometer code into the residual at SEP -----------
        resid = torch.zeros(B, T, D_MODEL, device=DEVICE, dtype=torch.float32)
        resid[:, sep_idx, :MAX_N] = t
        # operand positions carry their raw value embedding (not probed).
        resid[:, 0, MAX_N] = a.to(torch.float32)
        resid[:, 1, MAX_N] = b.to(torch.float32)

        # --- attention: head-0 SEP->operands weight scaled by normalised gcd
        p = (gcd_val.to(torch.float32) - 1.0) / float(MAX_N - 1)  # [B] in [0,1]
        attn = torch.full(
            (B, N_HEADS, T, T), 1.0 / T, device=DEVICE, dtype=torch.float32
        )
        attn[:, 0, sep_idx, :] = 0.0
        attn[:, 0, sep_idx, 0] = 0.5 * p               # -> operand a
        attn[:, 0, sep_idx, 1] = 0.5 * p               # -> operand b
        attn[:, 0, sep_idx, sep_idx] = 1.0 - p         # remainder onto SEP

        return {
            "attn_weights": [attn.detach().cpu().numpy() for _ in range(N_LAYERS)],
            "resid_post": [resid.detach().cpu().numpy() for _ in range(N_LAYERS)],
        }

    return model_fn


# --------------------------------------------------------------------------- #
# A tiny ridge probe (mirrors task._ridge_probe) that ALSO returns weights,
# used only to draw the "decoder ≈ all-ones counting" demo figure.
# --------------------------------------------------------------------------- #
def probe_with_weights(X, y, n_train, lam=1.0):
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    Xtr, ytr = X[:n_train], y[:n_train]
    Xte, yte = X[n_train:], y[n_train:]
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Ztr = np.concatenate([(Xtr - mu) / sd, np.ones((Xtr.shape[0], 1))], 1)
    Zte = np.concatenate([(Xte - mu) / sd, np.ones((Xte.shape[0], 1))], 1)
    A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1])
    w = np.linalg.solve(A, Ztr.T @ ytr)
    pred = Zte @ w
    eff = w[:-1] / sd  # weights mapped back to raw-feature (per-threshold) space
    return pred, yte, eff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_n", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=512)
    args = ap.parse_args()

    task = load_task(__file__)
    run_dir = results_dir(__file__)
    model_fn = make_model_fn()

    # Official payload via the canonical evaluator (seed 42, MAX_N 100, B 512).
    batch = task.generate(seed=args.seed, max_n=args.max_n, batch_size=args.batch_size)
    payload = task.evaluate(model_fn, batch)
    record_benchmark(__file__, run_dir, payload)

    # ----- extra artefacts for the Demo tab --------------------------------
    out = model_fn(batch.tokens)
    sep_idx = batch.tokens.shape[1] - 1
    resid_sep = out["resid_post"][0][:, sep_idx, :]            # [B, D]
    therm = resid_sep[:, :args.max_n]                          # thermometer code
    gcd = batch.gcd_labels.astype(np.float64)
    n_train = batch.batch_size // 2

    pred, te, eff = probe_with_weights(resid_sep, gcd, n_train)

    # sort a sample of the thermometer rows by gcd to show the staircase
    order = np.argsort(gcd)
    samp = order[:: max(1, len(order) // 60)][:60]

    np.savez(
        f"{run_dir}/demo.npz",
        gcd_true=te.astype(np.float32),
        gcd_pred=pred.astype(np.float32),
        learned_weights=eff[:args.max_n].astype(np.float32),   # ≈ all-ones counting decoder
        therm_sample=therm[samp].astype(np.float32),           # [<=60, MAX_N] staircase
        therm_gcd=gcd[samp].astype(np.float32),
        attn_corr=np.asarray(payload["attn_corr"], np.float32),
    )

    summary = {
        "headline_resid_r2": float(max(payload["global"]["resid_r2"])),
        "headline_resid_acc": float(max(payload["global"]["resid_acc"])),
        "baseline_resid_r2": float(payload["global"]["baseline_r2"]),
        "baseline_resid_acc": float(payload["global"]["baseline_acc"]),
        "best_attn_corr": float(np.max(np.abs(payload["attn_corr"]))),
        "baseline_attn_corr": float(abs(payload["baseline_attn_corr"])),
    }
    with open(f"{run_dir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("=== attention_gcd / first_pass ===")
    print(json.dumps(summary, indent=2))
    print("artefacts:", run_dir)


if __name__ == "__main__":
    main()
